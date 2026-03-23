# COLLEBORATION WITH LENNON

import cv2
import numpy as np
import base64
import os
import atexit
import time
import chess
from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI

# ============================================================
# Configuration
# ============================================================
BOARD_SIZE            = 320
CHESSBOARD_SIZE       = (7, 7)
TARGET_FPS            = 3
FRAME_DELAY           = 1.0 / TARGET_FPS

CENTER_CROP           = 0.35
OTSU_MIN_THRESH       = 15
FIXED_FALLBACK_THRESH = 25
OCC_RATIO_THRESH      = 0.25    # Tune this via /debug if shadow bleed occurs
STABILITY_MAX_PCT     = 2.5
MIN_FOOTPRINT_IOU     = 0.5     # Minimum intersection-over-union to accept a move
MAX_CHANGED_SQUARES   = 6       # More than this = lighting/shadow issue, reject early

# Rotation applied after perspective warp on every frame.
#   None                           = no rotation
#   cv2.ROTATE_90_CLOCKWISE        = rotate 90° right
#   cv2.ROTATE_90_COUNTERCLOCKWISE = rotate 90° left
#   cv2.ROTATE_180                 = flip upside down
BOARD_ROTATION = cv2.ROTATE_90_CLOCKWISE

# ============================================================
# Global State
# ============================================================
locked_outer_pts = None
last_stable_gray = None   # None = next /capture seeds; Set = next /capture detects
chess_board      = chess.Board()

virtual_board = [
    ['r','n','b','q','k','b','n','r'],
    ['p','p','p','p','p','p','p','p'],
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['P','P','P','P','P','P','P','P'],
    ['R','N','B','Q','K','B','N','R']
]

PIECE_MAP = {
    (chess.PAWN,   chess.WHITE): 'P', (chess.KNIGHT, chess.WHITE): 'N',
    (chess.BISHOP, chess.WHITE): 'B', (chess.ROOK,   chess.WHITE): 'R',
    (chess.QUEEN,  chess.WHITE): 'Q', (chess.KING,   chess.WHITE): 'K',
    (chess.PAWN,   chess.BLACK): 'p', (chess.KNIGHT, chess.BLACK): 'n',
    (chess.BISHOP, chess.BLACK): 'b', (chess.ROOK,   chess.BLACK): 'r',
    (chess.QUEEN,  chess.BLACK): 'q', (chess.KING,   chess.BLACK): 'k',
}

ui = WebUI()

# ============================================================
# Camera
# ============================================================
def initialize_camera():
    os.system("fuser -k /dev/video* 2>/dev/null")
    time.sleep(1)
    for index in [0, 1, 2, 3]:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 3)
                return cap
        except:
            continue
    return None

camera = initialize_camera()

@atexit.register
def shutdown_handler():
    if camera:
        camera.release()

# ============================================================
# Core Helpers
# ============================================================
def sync_virtual_board():
    global virtual_board
    for r in range(8):
        for c in range(8):
            sq    = chess.square(c, 7 - r)
            piece = chess_board.piece_at(sq)
            virtual_board[r][c] = PIECE_MAP[(piece.piece_type, piece.color)] if piece else '.'


def get_outer_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    grid     = refined.reshape(7, 7, 2)
    h_step   = np.mean(grid[:, 1:] - grid[:, :-1], axis=(0, 1))
    v_step   = np.mean(grid[1:, :] - grid[:-1, :], axis=(0, 1))
    return np.array([
        grid[0, 0] - h_step - v_step,
        grid[0, 6] + h_step - v_step,
        grid[6, 6] + h_step + v_step,
        grid[6, 0] - h_step + v_step
    ], dtype="float32")


def get_perspective_matrix():
    return cv2.getPerspectiveTransform(
        locked_outer_pts,
        np.array([
            [0,          0         ],
            [BOARD_SIZE, 0         ],
            [BOARD_SIZE, BOARD_SIZE],
            [0,          BOARD_SIZE]
        ], dtype="float32")
    )


def apply_warp(frame, M):
    """Warp + optional rotation. All frames go through here."""
    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))
    if BOARD_ROTATION is not None:
        warped = cv2.rotate(warped, BOARD_ROTATION)
    return warped


def encode_image(bgr_frame):
    _, buffer = cv2.imencode('.jpg', bgr_frame)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def capture_stable_median(M):
    """
    Flush stale V4L2 buffer, capture 3 frames, check stability.
    Returns (median_gray uint8, last_bgr, stability_pct).
    median_gray=None if unstable; stability_pct=None if camera failed.
    """
    time.sleep(0.5)
    for _ in range(2):
        camera.read()

    grays    = []
    last_bgr = None
    start    = time.time()

    for i in range(3):
        wait = (start + i * FRAME_DELAY) - time.time()
        if wait > 0:
            time.sleep(wait)
        success, frame = camera.read()
        if not success:
            return None, None, None
        warped   = apply_warp(frame, M)
        grays.append(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        last_bgr = warped

    stack         = np.stack([g.astype(np.float32) for g in grays], axis=0)
    stability_pct = float(round(np.mean(np.ptp(stack, axis=0)) / 255.0 * 100, 2))

    if stability_pct > STABILITY_MAX_PCT:
        return None, last_bgr, stability_pct

    median_gray = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    return median_gray, last_bgr, stability_pct


# ============================================================
# Corner Detection
# ============================================================
def try_find_corners(gray):
    """
    Multiple preprocessing variants → first success wins.
    Falls back to findChessboardCornersSB on low-contrast boards.
    """
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq      = clahe.apply(gray)
    eq_blur = cv2.GaussianBlur(eq, (5, 5), 0)
    norm    = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    adapt   = cv2.adaptiveThreshold(
                  gray, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY, 11, 2)
    attempts = [eq, eq_blur, norm, adapt, gray]
    flags    = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK)

    for img in attempts:
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, flags)
        if ret:
            return ret, corners

    for img in attempts:
        try:
            ret, corners = cv2.findChessboardCornersSB(img, CHESSBOARD_SIZE, 0)
            if ret:
                return ret, corners
        except:
            continue

    return False, None


# ============================================================
# Square Change Detection
# ============================================================
def get_square_roi_bounds(r, c):
    sq  = BOARD_SIZE // 8
    pad = int(sq * (1 - CENTER_CROP) / 2)
    return r*sq+pad, (r+1)*sq-pad, c*sq+pad, (c+1)*sq-pad


def square_changed(diff_img, r, c):
    """
    Returns (changed: bool, ratio: float) for square (r,c).
    Per-square Otsu with fixed fallback + morphological closing.
    """
    y1, y2, x1, x2 = get_square_roi_bounds(r, c)
    roi    = diff_img[y1:y2, x1:x2]
    area   = (y2-y1) * (x2-x1)
    if area == 0:
        return False, 0.0

    kernel           = np.ones((3, 3), np.uint8)
    otsu_val, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val < OTSU_MIN_THRESH:
        _, thresh = cv2.threshold(roi, FIXED_FALLBACK_THRESH, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    ratio  = cv2.countNonZero(thresh) / area
    return ratio > OCC_RATIO_THRESH, round(ratio, 3)


def get_changed_squares(diff_img):
    """
    Scans all 64 squares and returns a set of chess square indices
    where the diff exceeds OCC_RATIO_THRESH.
    Also returns a dict of {square_name: ratio} for all squares.
    """
    changed = set()
    ratios  = {}

    for r in range(8):
        for c in range(8):
            chess_sq       = chess.square(c, 7 - r)
            changed_flag, ratio = square_changed(diff_img, r, c)
            ratios[chess.square_name(chess_sq)] = ratio
            if changed_flag:
                changed.add(chess_sq)

    return changed, ratios


# ============================================================
# Move Footprint
# ============================================================
def get_move_footprint(move):
    """
    Returns the exact set of squares that physically change when this move is played.

    Quiet move / capture : {from, to}           — 2 squares
    En passant           : {from, to, ep_pawn}  — 3 squares
    Castling             : {king_from, king_to,
                            rook_from, rook_to} — 4 squares
    Promotion            : {from, to}           — 2 squares (piece type doesn't affect squares)
    """
    footprint = {move.from_square, move.to_square}

    if chess_board.is_en_passant(move):
        # Captured pawn sits on same file as to_square, same rank as from_square
        ep_sq = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square)
        )
        footprint.add(ep_sq)

    if chess_board.is_castling(move):
        # Add rook's from and to squares explicitly
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:  # Kingside
            footprint.add(chess.square(7, rank))    # Rook from (h-file)
            footprint.add(chess.square(5, rank))    # Rook to   (f-file)
        else:                                        # Queenside
            footprint.add(chess.square(0, rank))    # Rook from (a-file)
            footprint.add(chess.square(3, rank))    # Rook to   (d-file)

    return footprint


# ============================================================
# Move Detection — footprint-first approach
# ============================================================
def detect_move(diff_img):
    """
    The correct order:
      1. Detect which squares changed from the camera diff.
      2. For each legal move, compute its exact footprint.
      3. Score = IoU (intersection / union) of footprint vs changed squares.
      4. Accept the move with the best IoU — must exceed MIN_FOOTPRINT_IOU.

    Why IoU works:
      - Perfect quiet move: changed={e2,e4}, footprint={e2,e4} → IoU=1.0
      - Shadow bleed:       changed={e2,e4,d3}, footprint={e2,e4} → IoU=2/3=0.67
      - Castling (perfect): changed={e1,g1,h1,f1}, footprint={e1,g1,h1,f1} → IoU=1.0
      - En passant (perf.): changed={d5,e6,e5}, footprint={d5,e6,e5} → IoU=1.0
      - Wrong move:         changed={e2,e4}, footprint={d2,d4} → IoU=0/4=0.0
    """
    changed_squares, ratios = get_changed_squares(diff_img)

    if not changed_squares:
        return None, "No squares detected as changed. Did you make a move?", set(), ratios

    if len(changed_squares) > MAX_CHANGED_SQUARES:
        names = sorted([chess.square_name(s) for s in changed_squares])
        return None, (
            f"Too many squares changed ({len(changed_squares)}): {names}. "
            "Shadow or lighting issue — raise OCC_RATIO_THRESH in config."
        ), changed_squares, ratios

    best_iou   = -1.0
    best_moves = []

    for move in chess_board.legal_moves:
        footprint    = get_move_footprint(move)
        intersection = len(footprint & changed_squares)
        union        = len(footprint | changed_squares)
        iou          = intersection / union if union > 0 else 0.0

        if iou > best_iou:
            best_iou   = iou
            best_moves = [move]
        elif iou == best_iou:
            best_moves.append(move)

    if not best_moves:
        return None, "No legal moves available — game may be over.", changed_squares, ratios

    # Among ties (promotion variants) prefer queen
    best_move = next(
        (m for m in best_moves if m.promotion == chess.QUEEN),
        best_moves[0]
    )

    changed_names = sorted([chess.square_name(s) for s in changed_squares])
    expected_names = sorted([chess.square_name(s) for s in get_move_footprint(best_move)])

    if best_iou < MIN_FOOTPRINT_IOU:
        return None, (
            f"No legal move matches changed squares {changed_names} "
            f"(best IoU: {best_iou:.0%}, expected footprint: {expected_names}). "
            "Call /debug for details."
        ), changed_squares, ratios

    return best_move, chess_board.san(best_move), changed_squares, ratios


# ============================================================
# API: Stream
# ============================================================
def get_live_frame():
    if not camera:
        return {"error": "No camera found."}
    success, frame = camera.read()
    if not success:
        return {"error": "Camera read failed."}
    preview = cv2.resize(frame, (320, 240))
    _, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return {"image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"}


# ============================================================
# API: Calibrate
# ============================================================
def calibrate_board():
    """
    Tries up to 10 frames with 5 preprocessing variants each.
    Call with the board EMPTY (no pieces).
    On persistent failure use /calibrate_manual.
    """
    global locked_outer_pts, last_stable_gray

    if not camera:
        return {"status": "error", "message": "No camera available."}

    for attempt in range(10):
        success, frame = camera.read()
        if not success:
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = try_find_corners(gray)

        if ret:
            locked_outer_pts = get_outer_corners(gray, corners)
            last_stable_gray = None
            return {
                "status":  "success",
                "message": (
                    f"Calibrated on attempt {attempt + 1}. "
                    "Call /verify_raw then /verify to confirm orientation."
                )
            }
        time.sleep(0.2)

    return {
        "status":  "fail",
        "message": (
            "Could not detect board after 10 attempts. "
            "Use /calibrate_manual with pixel coordinates from /stream."
        )
    }


# ============================================================
# API: Manual Calibration Fallback
# ============================================================
def calibrate_manual(tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y):
    """
    Set 4 outer board corners manually in raw camera pixel coordinates.
    /stream returns a 320x240 preview — multiply clicked coords by 2
    since the actual camera resolution is 640x480.

    TL=top-left, TR=top-right, BR=bottom-right, BL=bottom-left
    (from camera perspective, not chess orientation).

    Example:
    /calibrate_manual?tl_x=90&tl_y=60&tr_x=590&tr_y=60&br_x=590&br_y=460&bl_x=90&bl_y=460
    """
    global locked_outer_pts, last_stable_gray
    try:
        locked_outer_pts = np.array([
            [float(tl_x), float(tl_y)],
            [float(tr_x), float(tr_y)],
            [float(br_x), float(br_y)],
            [float(bl_x), float(bl_y)],
        ], dtype="float32")
        last_stable_gray = None
        return {
            "status":  "success",
            "message": "Manual corners set. Call /verify_raw then /verify."
        }
    except Exception as e:
        return {"status": "error", "message": f"Invalid coordinates: {e}"}


# ============================================================
# API: Verify Raw (pre-warp corner dots)
# ============================================================
def verify_raw():
    """
    Shows raw camera frame with locked_outer_pts drawn as colored dots.
    Confirms the corners sit on actual board edges before the warp.

    GREEN=a1/BL, RED=a8/TL, CYAN=h8/TR, BLUE=h1/BR
    If dots are off, use /calibrate_manual with corrected coords.
    """
    if locked_outer_pts is None:
        return {"status": "fail", "message": "Calibrate first."}

    success, frame = camera.read()
    if not success:
        return {"status": "fail", "message": "Camera read failed."}

    vis    = frame.copy()
    labels = ["TL→a8", "TR→h8", "BR→h1", "BL→a1"]
    colors = [(0,0,255), (255,255,0), (255,0,0), (0,255,0)]

    for pt, label, color in zip(locked_outer_pts, labels, colors):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.putText(vis, label, (x+10, y+5),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2, cv2.LINE_AA)

    pts = locked_outer_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (255, 255, 255), 2)

    return {
        "status": "ok",
        "image":  encode_image(cv2.resize(vis, (480, 360))),
        "note":   "GREEN=a1, RED=a8, CYAN=h8, BLUE=h1. If dots are off the board use /calibrate_manual."
    }


# ============================================================
# API: Verify Orientation (post-warp labels)
# ============================================================
def verify_calibration():
    """
    Warps current frame and draws square labels + piece symbols.
    Cyan boxes show the center-crop zone sampled per square.

    GREEN=a1 (white bottom-left), BLUE=h1, RED=a8, YELLOW=h8.
    If rotated, change BOARD_ROTATION in config and recalibrate.
    """
    if locked_outer_pts is None:
        return {"status": "fail", "message": "Calibrate first."}

    success, frame = camera.read()
    if not success:
        return {"status": "fail", "message": "Camera read failed."}

    M      = get_perspective_matrix()
    warped = apply_warp(frame, M)
    vis    = warped.copy()
    sq     = BOARD_SIZE // 8
    pad    = int(sq * (1 - CENTER_CROP) / 2)

    corner_colors = {
        (7,0): (0,255,0),    # a1 green
        (7,7): (255,100,0),  # h1 blue
        (0,0): (0,0,255),    # a8 red
        (0,7): (0,255,255),  # h8 yellow
    }

    for r in range(8):
        for c in range(8):
            x1, y1 = c*sq, r*sq
            x2, y2 = x1+sq, y1+sq
            color  = corner_colors.get((r,c), (255,255,255))
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 1)
            cv2.rectangle(vis, (x1+pad,y1+pad), (x2-pad,y2-pad), (255,255,0), 1)

            chess_sq   = chess.square(c, 7-r)
            label      = chess.square_name(chess_sq)
            piece      = chess_board.piece_at(chess_sq)
            piece_char = piece.symbol() if piece else ""
            cx, cy     = x1+sq//2, y1+sq//2

            cv2.putText(vis, label, (cx-10,cy-2),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, color, 1, cv2.LINE_AA)
            if piece_char:
                cv2.putText(vis, piece_char, (cx-4,cy+10),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, (0,255,0), 1, cv2.LINE_AA)

    return {
        "status": "ok",
        "image":  encode_image(vis),
        "note":   "GREEN=a1 bottom-left. Cyan=sampled zone. If rotated change BOARD_ROTATION."
    }


# ============================================================
# API: Debug (text-only)
# ============================================================
def debug_capture():
    """
    Text-only debug — readable directly in the browser as JSON.
    Call AFTER /capture (reference seeded) and AFTER making a move,
    BEFORE pressing /capture again.

    ratio_grid:        8x8 diff ratios — moved squares should be >>0.4
    changed_grid:      8x8 — X=flagged changed, .=not changed
    known_grid:        8x8 — current piece layout from chess_board
    changed_squares:   list of square names detected as changed
    footprint_matches: top 5 legal moves with IoU scores
    threshold:         current OCC_RATIO_THRESH
    """
    if locked_outer_pts is None:
        return {"status": "fail", "message": "Calibrate first."}
    if last_stable_gray is None:
        return {"status": "fail", "message": "Call /capture once first to seed reference."}

    M = get_perspective_matrix()

    success, frame = camera.read()
    if not success:
        return {"status": "fail", "message": "Camera read failed."}

    warped    = apply_warp(frame, M)
    curr_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    diff_raw  = cv2.absdiff(curr_gray, last_stable_gray)

    changed_squares, ratios = get_changed_squares(diff_raw)
    changed_names           = sorted([chess.square_name(s) for s in changed_squares])

    # Build display grids
    ratio_rows   = []
    changed_rows = []
    known_rows   = []

    for r in range(8):
        ratio_row   = []
        changed_row = []
        known_row   = []
        for c in range(8):
            chess_sq = chess.square(c, 7-r)
            sq_name  = chess.square_name(chess_sq)
            piece    = chess_board.piece_at(chess_sq)
            ratio_row.append(f"{ratios.get(sq_name, 0):.2f}")
            changed_row.append("X" if chess_sq in changed_squares else ".")
            known_row.append(piece.symbol() if piece else ".")
        ratio_rows.append(ratio_row)
        changed_rows.append(changed_row)
        known_rows.append(known_row)

    # Score all legal moves by IoU against detected changed squares
    scored = []
    for move in chess_board.legal_moves:
        footprint    = get_move_footprint(move)
        intersection = len(footprint & changed_squares)
        union        = len(footprint | changed_squares)
        iou          = intersection / union if union > 0 else 0.0
        fp_names     = sorted([chess.square_name(s) for s in footprint])
        scored.append((iou, chess_board.san(move), fp_names))

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = [
        {
            "move":      s[1],
            "iou":       f"{s[0]:.0%}",
            "footprint": s[2]
        }
        for s in scored[:5]
    ]

    return {
        "status":           "debug",
        "threshold":        OCC_RATIO_THRESH,
        "min_iou_required": MIN_FOOTPRINT_IOU,
        "changed_squares":  changed_names,
        "ratio_grid":       ratio_rows,
        "changed_grid":     changed_rows,
        "known_grid":       known_rows,
        "footprint_matches": top5
    }


# ============================================================
# API: Capture & Analyze
# ============================================================
def capture_and_analyze():
    """
    One press per turn.

    First call  → seeds reference frame, returns "ready".
    Every after → detects changed squares, matches to legal move footprint,
                  accepts if IoU >= MIN_FOOTPRINT_IOU.
    On failure  → reference unchanged, player retries without re-seeding.
    """
    global chess_board, last_stable_gray

    if locked_outer_pts is None:
        return {"status": "fail", "message": "Please calibrate the board first."}
    if not camera:
        return {"status": "fail", "message": "No camera available."}

    M = get_perspective_matrix()
    median_gray, last_bgr, stability_pct = capture_stable_median(M)

    if stability_pct is None:
        return {"status": "fail", "message": "Camera read failed. Check connection."}

    if median_gray is None:
        return {
            "status":          "unstable",
            "message":         f"Not stable ({stability_pct:.2f}%). Hold still and try again.",
            "stability_score": stability_pct,
            "image":           encode_image(last_bgr) if last_bgr is not None else None
        }

    # ── First call: seed reference ────────────────────────────────────────────
    if last_stable_gray is None:
        last_stable_gray = median_gray
        return {
            "status":          "ready",
            "message":         "Reference saved. Make your move and press Capture.",
            "fen":             chess_board.fen(),
            "stability_score": stability_pct,
            "image":           encode_image(last_bgr)
        }

    # ── Subsequent calls: detect via footprint ────────────────────────────────
    diff_img           = cv2.absdiff(median_gray, last_stable_gray)
    move, msg, changed, ratios = detect_move(diff_img)

    changed_names = sorted([chess.square_name(s) for s in changed])

    if move is None:
        return {
            "status":          "fail",
            "message":         msg,
            "changed_squares": changed_names,
            "fen":             chess_board.fen(),
            "stability_score": stability_pct,
            "image":           encode_image(last_bgr)
        }

    chess_board.push(move)
    sync_virtual_board()
    last_stable_gray = median_gray

    turn_label = "White" if chess_board.turn == chess.WHITE else "Black"
    return {
        "status":          "success",
        "message":         f"Move: {msg}. Now {turn_label}'s turn.",
        "changed_squares": changed_names,
        "fen":             chess_board.fen(),
        "stability_score": stability_pct,
        "image":           encode_image(last_bgr)
    }


# ============================================================
# API Registration
# ============================================================
ui.expose_api("GET", "/stream",           get_live_frame)
ui.expose_api("GET", "/calibrate",        calibrate_board)
ui.expose_api("GET", "/calibrate_manual", calibrate_manual)
ui.expose_api("GET", "/verify_raw",       verify_raw)
ui.expose_api("GET", "/verify",           verify_calibration)
ui.expose_api("GET", "/capture",          capture_and_analyze)
ui.expose_api("GET", "/debug",            debug_capture)

print(f"Server starting. URL: {ui.local_url}")
App.run()