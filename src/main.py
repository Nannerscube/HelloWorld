import atexit
import os
import time

import chess
import chess.engine
import cv2
import numpy as np
from arduino.app_utils import App, Bridge

print("main.py started")

SKILL_LEVEL = 10
ENGINE_PATH_CANDIDATES = [
    os.environ.get("STOCKFISH_PATH", ""),
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "stockfish",
]

BOARD_SIZE = 320
CHESSBOARD_SIZE = (7, 7)
TARGET_FPS = 3
FRAME_DELAY = 1.0 / TARGET_FPS
CENTER_CROP = 0.5
OTSU_MIN_THRESH = 15
FIXED_FALLBACK_THRESH = 25
OCC_RATIO_THRESH = 0.25
OCC_RATIO_MIN_THRESH = 0.16
OCC_RATIO_MAX_THRESH = 0.42
OCC_RATIO_BASELINE = 0.18
OCC_RATIO_NOISE_SCALE = 2.0
OCC_RATIO_MARGIN = 0.03
STABILITY_MAX_PCT = 2.5
MIN_FOOTPRINT_IOU = 0.5
MAX_CHANGED_SQUARES = 6
MAX_CAPTURE_CHANGED_SQUARES = 8
MOVE_SCORE_MIN = 1.35
CAPTURE_MOVE_SCORE_MIN = 1.15
CAPTURE_TO_RATIO_FLOOR = 0.08
CAPTURE_SOURCE_BONUS = 0.45
QUIET_FROM_CAPTURE_SOURCE_PENALTY = 0.5
BOARD_ROTATION = None
ROTATION_CANDIDATES = [
    ("none", None),
    ("rotate_90_cw", cv2.ROTATE_90_CLOCKWISE),
    ("rotate_180", cv2.ROTATE_180),
    ("rotate_90_ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
]
RED_AREA_MIN = 500

board = chess.Board()
engine = None
camera = None
locked_outer_pts = None
last_stable_gray = None
square_occ_thresholds = {}
camera_status = "Camera setup not started."
board_rotation = BOARD_ROTATION


def set_camera_status(message: str) -> None:
    global camera_status
    camera_status = message
    print(f"[VISION]: {message}")


def debug_log(message: str) -> None:
    print(f"[VISION DEBUG]: {message}")


def log_board_snapshot(label: str) -> None:
    turn = "white" if board.turn == chess.WHITE else "black"
    debug_log(f"{label} | fen={board.fen()} | turn={turn} | legal_count={board.legal_moves.count()}")


def initialize_engine() -> bool:
    global engine

    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_stockfish = os.path.join(repo_root, "stockfish")
    candidates = [repo_stockfish, *ENGINE_PATH_CANDIDATES]

    for candidate in candidates:
        if not candidate:
            continue
        exists = os.path.isfile(candidate)
        executable = os.access(candidate, os.X_OK) if exists else False
        print(f"[PYTHON]: Trying {candidate} | exists={exists} | executable={executable}")
        try:
            engine = chess.engine.SimpleEngine.popen_uci(candidate)
            engine.configure({"Skill Level": SKILL_LEVEL})
            print(f"[PYTHON]: Stockfish ready from {candidate}")
            debug_log(f"Engine initialized with starting board FEN: {board.fen()}")
            return True
        except Exception as exc:
            print(f"[PYTHON]: Failed ({candidate}): {type(exc).__name__}: {exc}")
            continue
    print("[PYTHON ERROR]: Could not initialize Stockfish from configured paths.")
    return False


def close_engine() -> None:
    global engine
    if engine is not None:
        engine.quit()
        engine = None
        print("[PYTHON]: Engine closed")


def initialize_camera():
    os.system("fuser -k /dev/video* 2>/dev/null")
    time.sleep(1)
    for index in [0, 1, 2, 3]:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                print(f"[VISION]: Camera opened at index {index}")
                return cap
        except Exception:
            continue
    print("[VISION ERROR]: No camera available")
    return None


def close_camera() -> None:
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("[VISION]: Camera released")


@atexit.register
def shutdown_handler() -> None:
    close_engine()
    close_camera()


def ensure_camera() -> bool:
    global camera
    if camera is None:
        camera = initialize_camera()
    return camera is not None


def format_result(status: str, message: str) -> str:
    return f"{status}:{message}"


def get_outer_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    grid = refined.reshape(7, 7, 2)
    h_step = np.mean(grid[:, 1:] - grid[:, :-1], axis=(0, 1))
    v_step = np.mean(grid[1:, :] - grid[:-1, :], axis=(0, 1))
    return np.array(
        [
            grid[0, 0] - h_step - v_step,
            grid[0, 6] + h_step - v_step,
            grid[6, 6] + h_step + v_step,
            grid[6, 0] - h_step + v_step,
        ],
        dtype="float32",
    )


def get_perspective_matrix():
    return cv2.getPerspectiveTransform(
        locked_outer_pts,
        np.array(
            [[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]],
            dtype="float32",
        ),
    )


def apply_warp(frame, matrix):
    warped = cv2.warpPerspective(frame, matrix, (BOARD_SIZE, BOARD_SIZE))
    if board_rotation is not None:
        warped = cv2.rotate(warped, board_rotation)
    return warped


def rotate_image(img, rotation):
    if rotation is None:
        return img
    return cv2.rotate(img, rotation)


def get_square_center_roi(gray_img, row, col, crop_ratio=0.45):
    square_size = BOARD_SIZE // 8
    inset = int(square_size * (1 - crop_ratio) / 2)
    y1 = row * square_size + inset
    y2 = (row + 1) * square_size - inset
    x1 = col * square_size + inset
    x2 = (col + 1) * square_size - inset
    return gray_img[y1:y2, x1:x2]


def score_starting_rotation(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    top_samples = []
    bottom_samples = []
    middle_samples = []
    top_texture = []
    bottom_texture = []
    middle_texture = []

    for row in [0, 1]:
        for col in range(8):
            roi = get_square_center_roi(gray, row, col)
            if roi.size:
                top_samples.append(float(np.mean(roi)))
                top_texture.append(float(np.std(roi)))

    for row in [6, 7]:
        for col in range(8):
            roi = get_square_center_roi(gray, row, col)
            if roi.size:
                bottom_samples.append(float(np.mean(roi)))
                bottom_texture.append(float(np.std(roi)))

    for row in [2, 3, 4, 5]:
        for col in range(8):
            roi = get_square_center_roi(gray, row, col)
            if roi.size:
                middle_samples.append(float(np.mean(roi)))
                middle_texture.append(float(np.std(roi)))

    top_mean = float(np.mean(top_samples)) if top_samples else 0.0
    bottom_mean = float(np.mean(bottom_samples)) if bottom_samples else 0.0
    middle_mean = float(np.mean(middle_samples)) if middle_samples else 0.0
    top_std = float(np.mean(top_texture)) if top_texture else 0.0
    bottom_std = float(np.mean(bottom_texture)) if bottom_texture else 0.0
    middle_std = float(np.mean(middle_texture)) if middle_texture else 0.0

    # Correct starting orientation should show dark black pieces on the top ranks,
    # lighter white pieces on the bottom ranks, and more texture on occupied ranks
    # than the empty middle ranks. A 90-degree rotation tends to flatten this signal.
    vertical_contrast = bottom_mean - top_mean
    occupied_texture = ((top_std + bottom_std) / 2.0) - middle_std
    edge_penalty = abs(middle_mean - ((top_mean + bottom_mean) / 2.0))
    score = vertical_contrast * 1.5 + occupied_texture - edge_penalty * 0.2

    return {
        "score": round(score, 3),
        "top_mean": round(top_mean, 2),
        "bottom_mean": round(bottom_mean, 2),
        "middle_mean": round(middle_mean, 2),
        "top_std": round(top_std, 2),
        "bottom_std": round(bottom_std, 2),
        "middle_std": round(middle_std, 2),
    }


def detect_starting_rotation(warped_bgr):
    best_name = "none"
    best_rotation = None
    best_score = None

    for name, rotation in ROTATION_CANDIDATES:
        rotated = rotate_image(warped_bgr, rotation)
        metrics = score_starting_rotation(rotated)
        debug_log(
            "Orientation probe "
            f"{name}: score={metrics['score']:.3f}, "
            f"top_mean={metrics['top_mean']:.2f}, bottom_mean={metrics['bottom_mean']:.2f}, "
            f"top_std={metrics['top_std']:.2f}, bottom_std={metrics['bottom_std']:.2f}, "
            f"middle_std={metrics['middle_std']:.2f}"
        )
        if best_score is None or metrics["score"] > best_score:
            best_score = metrics["score"]
            best_name = name
            best_rotation = rotation

    return best_name, best_rotation, best_score


def rotation_name(rotation):
    for name, candidate in ROTATION_CANDIDATES:
        if candidate == rotation:
            return name
    return "unknown"


def get_red_anchor_side(raw_bgr):
    if raw_bgr is None or locked_outer_pts is None:
        return None, None

    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > best_area:
            best_area = area
            best_contour = contour

    if best_contour is None or best_area < RED_AREA_MIN:
        return None, None

    moments = cv2.moments(best_contour)
    if moments["m00"] == 0:
        return None, None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    anchor = np.array([cx, cy], dtype=np.float32)

    side_midpoints = {
        "top": (locked_outer_pts[0] + locked_outer_pts[1]) / 2.0,
        "right": (locked_outer_pts[1] + locked_outer_pts[2]) / 2.0,
        "bottom": (locked_outer_pts[2] + locked_outer_pts[3]) / 2.0,
        "left": (locked_outer_pts[3] + locked_outer_pts[0]) / 2.0,
    }

    side_distances = {
        side: float(np.linalg.norm(anchor - midpoint))
        for side, midpoint in side_midpoints.items()
    }
    detected_side = min(side_distances, key=side_distances.get)
    return detected_side, {
        "area": round(best_area, 1),
        "cx": round(float(cx), 1),
        "cy": round(float(cy), 1),
        "distances": {side: round(dist, 1) for side, dist in side_distances.items()},
    }


def get_rotation_for_bottom_side(side_name):
    return {
        "bottom": None,
        "top": cv2.ROTATE_180,
        "left": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "right": cv2.ROTATE_90_CLOCKWISE,
    }.get(side_name)


def choose_starting_rotation(raw_bgr, warped_bgr):
    red_side, red_metrics = get_red_anchor_side(raw_bgr)
    if red_side is not None:
        chosen_rotation = get_rotation_for_bottom_side(red_side)
        debug_log(
            "Red anchor probe: "
            f"side={red_side}, area={red_metrics['area']}, center=({red_metrics['cx']}, {red_metrics['cy']}), "
            f"distances={red_metrics['distances']}"
        )
        return rotation_name(chosen_rotation), chosen_rotation, "red-anchor"

    best_name, best_rotation, best_score = detect_starting_rotation(warped_bgr)
    debug_log(
        f"Red anchor unavailable; falling back to board heuristic rotation {best_name} (score={best_score:.3f})."
    )
    return best_name, best_rotation, "board-heuristic"


def capture_stable_median(matrix):
    debug_log("Starting stable median capture.")
    time.sleep(0.5)
    for _ in range(2):
        camera.read()

    grays = []
    last_bgr = None
    last_raw = None
    start = time.time()
    for i in range(3):
        wait = (start + i * FRAME_DELAY) - time.time()
        if wait > 0:
            time.sleep(wait)
        success, frame = camera.read()
        if not success:
            debug_log("Camera read failed during stable median capture.")
            return None, None, None, None, None
        last_raw = frame.copy()
        warped = apply_warp(frame, matrix)
        grays.append(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        last_bgr = warped

    stack = np.stack([g.astype(np.float32) for g in grays], axis=0)
    stability_pct = float(round(np.mean(np.ptp(stack, axis=0)) / 255.0 * 100, 2))
    debug_log(f"Stable median capture finished with stability={stability_pct:.2f}%")
    if stability_pct > STABILITY_MAX_PCT:
        debug_log("Stable median capture rejected as unstable.")
        return None, last_bgr, stability_pct, grays, last_raw

    median_gray = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    debug_log("Stable median capture accepted.")
    return median_gray, last_bgr, stability_pct, grays, last_raw


def try_find_corners(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq_blur = cv2.GaussianBlur(eq, (5, 5), 0)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    attempts = [eq, eq_blur, norm, adapt, gray]
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    for img in attempts:
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, flags)
        if ret:
            return ret, corners

    for img in attempts:
        try:
            ret, corners = cv2.findChessboardCornersSB(img, CHESSBOARD_SIZE, 0)
            if ret:
                return ret, corners
        except Exception:
            continue

    return False, None


def get_square_roi_bounds(row, col):
    square_size = BOARD_SIZE // 8
    pad = int(square_size * (1 - CENTER_CROP) / 2)
    return row * square_size + pad, (row + 1) * square_size - pad, col * square_size + pad, (col + 1) * square_size - pad


def measure_change_ratio(roi):
    area = roi.shape[0] * roi.shape[1]
    if area == 0:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    otsu_val, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val < OTSU_MIN_THRESH:
        _, thresh = cv2.threshold(roi, FIXED_FALLBACK_THRESH, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return round(cv2.countNonZero(thresh) / area, 3)


def compute_square_thresholds(gray_frames, reference_gray):
    thresholds = {}
    if not gray_frames or reference_gray is None:
        return thresholds

    for row in range(8):
        for col in range(8):
            y1, y2, x1, x2 = get_square_roi_bounds(row, col)
            noise_ratios = []
            reference_roi = reference_gray[y1:y2, x1:x2]
            for gray in gray_frames:
                diff_roi = cv2.absdiff(gray[y1:y2, x1:x2], reference_roi)
                noise_ratios.append(measure_change_ratio(diff_roi))

            noise_peak = max(noise_ratios) if noise_ratios else 0.0
            learned = OCC_RATIO_BASELINE + noise_peak * OCC_RATIO_NOISE_SCALE + OCC_RATIO_MARGIN
            learned = min(max(learned, OCC_RATIO_MIN_THRESH), OCC_RATIO_MAX_THRESH)
            thresholds[chess.square(col, 7 - row)] = round(learned, 3)

    return thresholds


def update_square_thresholds(gray_frames, reference_gray):
    global square_occ_thresholds
    square_occ_thresholds = compute_square_thresholds(gray_frames, reference_gray)
    if not square_occ_thresholds:
        debug_log("Per-square thresholds unavailable, keeping global fallback only.")
        return

    sorted_thresholds = sorted(
        ((chess.square_name(square), threshold) for square, threshold in square_occ_thresholds.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    debug_log(f"Per-square thresholds learned: {sorted_thresholds}")


def square_changed(diff_img, row, col):
    y1, y2, x1, x2 = get_square_roi_bounds(row, col)
    roi = diff_img[y1:y2, x1:x2]
    ratio = measure_change_ratio(roi)
    chess_sq = chess.square(col, 7 - row)
    threshold = square_occ_thresholds.get(chess_sq, OCC_RATIO_THRESH)
    return ratio > threshold, ratio


def get_changed_squares(diff_img):
    changed = set()
    ratios = {}
    for row in range(8):
        for col in range(8):
            chess_sq = chess.square(col, 7 - row)
            changed_flag, ratio = square_changed(diff_img, row, col)
            ratios[chess.square_name(chess_sq)] = ratio
            if changed_flag:
                changed.add(chess_sq)
    return changed, ratios


def square_distance(square_a, square_b):
    return max(
        abs(chess.square_file(square_a) - chess.square_file(square_b)),
        abs(chess.square_rank(square_a) - chess.square_rank(square_b)),
    )


def score_move_candidate(move, changed_squares, ratios):
    footprint = get_move_footprint(move)
    footprint_names = {chess.square_name(square) for square in footprint}
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)
    from_ratio = ratios.get(from_name, 0.0)
    to_ratio = ratios.get(to_name, 0.0)

    intersection = len(footprint & changed_squares)
    union = len(footprint | changed_squares)
    iou = intersection / union if union > 0 else 0.0

    extra_squares = changed_squares - footprint
    local_spill_penalty = 0.0
    far_spill_penalty = 0.0
    for square in extra_squares:
        ratio = ratios.get(chess.square_name(square), 0.0)
        nearest = min(square_distance(square, fp_square) for fp_square in footprint)
        if nearest <= 1:
            local_spill_penalty += ratio * 0.35
        else:
            far_spill_penalty += ratio * 1.1

    is_capture = board.is_capture(move)
    from_weight = 1.7
    to_weight = 1.6 if is_capture else 1.9
    score = iou * 1.8 + from_ratio * from_weight + to_ratio * to_weight

    if is_capture:
        score += 0.2 + CAPTURE_SOURCE_BONUS * min(from_ratio, 0.6)
        if to_ratio >= CAPTURE_TO_RATIO_FLOOR:
            score += 0.2
        else:
            score -= (CAPTURE_TO_RATIO_FLOOR - to_ratio) * 2.5

    score -= local_spill_penalty
    score -= far_spill_penalty

    return {
        "move": move,
        "score": round(score, 3),
        "iou": round(iou, 3),
        "from_ratio": round(from_ratio, 3),
        "to_ratio": round(to_ratio, 3),
        "is_capture": is_capture,
        "extra_squares": sorted(chess.square_name(square) for square in extra_squares),
        "footprint": sorted(footprint_names),
    }


def get_move_footprint(move):
    footprint = {move.from_square, move.to_square}
    if board.is_en_passant(move):
        ep_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        footprint.add(ep_sq)
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) == 6:
            footprint.add(chess.square(7, rank))
            footprint.add(chess.square(5, rank))
        else:
            footprint.add(chess.square(0, rank))
            footprint.add(chess.square(3, rank))
    return footprint


def detect_move(diff_img):
    changed_squares, ratios = get_changed_squares(diff_img)
    sorted_ratios = sorted(ratios.items(), key=lambda item: item[1], reverse=True)[:12]
    debug_log(f"Top changed-square ratios: {sorted_ratios}")
    if not changed_squares:
        debug_log("No changed squares detected.")
        return None, "No squares changed.", set(), ratios
    if len(changed_squares) > MAX_CAPTURE_CHANGED_SQUARES:
        names = sorted(chess.square_name(s) for s in changed_squares)
        debug_log(f"Too many changed squares detected: {names}")
        return None, f"Too many squares changed: {names}", changed_squares, ratios

    scored_moves = []
    capture_sources = set()
    for move in board.legal_moves:
        if board.is_capture(move):
            capture_sources.add(move.from_square)
        scored_moves.append(score_move_candidate(move, changed_squares, ratios))

    for candidate in scored_moves:
        if candidate["is_capture"]:
            continue
        if candidate["move"].from_square in capture_sources:
            candidate["score"] = round(
                candidate["score"] - QUIET_FROM_CAPTURE_SOURCE_PENALTY * min(candidate["from_ratio"], 0.8),
                3,
            )

    if not scored_moves:
        debug_log("No legal moves available while trying to detect move.")
        return None, "No legal moves available.", changed_squares, ratios

    scored_moves.sort(
        key=lambda item: (
            item["score"],
            item["iou"],
            item["to_ratio"],
            1 if item["move"].promotion == chess.QUEEN else 0,
        ),
        reverse=True,
    )
    top_candidates = [
        (
            candidate["move"].uci(),
            candidate["score"],
            candidate["iou"],
            candidate["from_ratio"],
            candidate["to_ratio"],
            candidate["extra_squares"],
        )
        for candidate in scored_moves[:5]
    ]
    debug_log(f"Top move candidates: {top_candidates}")

    best_candidate = scored_moves[0]
    best_move = best_candidate["move"]
    best_iou = best_candidate["iou"]
    best_score = best_candidate["score"]
    changed_names = sorted(chess.square_name(s) for s in changed_squares)
    best_footprint = best_candidate["footprint"]
    debug_log(f"Detected changed squares: {changed_names}")
    debug_log(
        f"Best move candidate: {best_move.uci()} with footprint {best_footprint}, "
        f"IoU {best_iou:.2f}, score {best_score:.2f}, capture={best_candidate['is_capture']}"
    )

    move_score_min = CAPTURE_MOVE_SCORE_MIN if best_candidate["is_capture"] else MOVE_SCORE_MIN
    changed_limit = MAX_CAPTURE_CHANGED_SQUARES if best_candidate["is_capture"] else MAX_CHANGED_SQUARES
    if len(changed_squares) > changed_limit:
        return None, f"Too many squares changed for move match: {changed_names}", changed_squares, ratios

    if best_score < move_score_min or (best_iou < MIN_FOOTPRINT_IOU and best_score < (move_score_min + 0.25)):
        return None, (
            f"No legal move match for {changed_names} "
            f"(IoU {best_iou:.0%}, score {best_score:.2f})"
        ), changed_squares, ratios
    return best_move, board.san(best_move), changed_squares, ratios


def uci_to_piece_type(square_str: str) -> int:
    square = chess.parse_square(square_str)
    piece = board.piece_at(square)
    if piece is None:
        return 0
    if piece.piece_type == chess.PAWN:
        return 1
    if piece.piece_type == chess.KNIGHT:
        return 3
    if piece.piece_type == chess.BISHOP:
        return 4
    if piece.piece_type == chess.ROOK:
        return 5
    if piece.piece_type == chess.QUEEN:
        return 6
    return 2


def get_robot_move() -> str:
    global board
    log_board_snapshot("get_robot_move entry")
    debug_log(f"get_robot_move called with board FEN before engine move: {board.fen()}")
    debug_log(f"Legal moves before engine move: {[move.uci() for move in list(board.legal_moves)[:20]]}")
    if engine is None:
        print("[PYTHON ERROR]: Engine not initialized")
        debug_log("Engine missing, returning fallback move e2,e4,1,0")
        return "e2,e4,1,0"
    try:
        result = engine.play(board, chess.engine.Limit(time=2.0))
        move = result.move
        debug_log(f"Engine selected move: {move.uci()}")
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        moving_piece_id = uci_to_piece_type(from_square)
        captured_piece_id = uci_to_piece_type(to_square)
        debug_log(
            f"Robot move details before push: from={from_square}, to={to_square}, moving_piece_id={moving_piece_id}, captured_piece_id={captured_piece_id}"
        )
        board.push(move)
        move_str = f"{from_square},{to_square},{moving_piece_id},{captured_piece_id}"
        print(f"[PYTHON]: Robot move -> {move_str}")
        print(f"[PYTHON]: Board after robot move: {board.fen()}")
        debug_log(f"Robot move pushed successfully. New FEN: {board.fen()}")
        log_board_snapshot("get_robot_move exit")
        return move_str
    except Exception as exc:
        print(f"[PYTHON ERROR]: Failed to get move: {exc}")
        debug_log(f"Engine move failed on board FEN {board.fen()}")
        debug_log("Returning fallback move e2,e4,1,0")
        log_board_snapshot("get_robot_move exception")
        return "e2,e4,1,0"


def reset_game() -> bool:
    global board
    log_board_snapshot("reset_game before")
    board = chess.Board()
    print("[PYTHON]: Game reset to starting position without clearing camera reference")
    debug_log(f"reset_game set board to FEN: {board.fen()}")
    log_board_snapshot("reset_game after")
    return True


def get_board_state() -> str:
    fen = board.fen()
    print(f"[PYTHON]: Current FEN: {fen}")
    debug_log(f"get_board_state returning FEN: {fen}")
    log_board_snapshot("get_board_state snapshot")
    return fen


def is_game_over() -> str:
    if board.is_checkmate():
        winner = "Player" if board.turn == chess.BLACK else "Robot"
        return f"Checkmate! {winner} wins!"
    if board.is_stalemate():
        return "Stalemate - Draw!"
    if board.is_insufficient_material():
        return "Draw - Insufficient material"
    if board.is_fifty_moves():
        return "Draw - Fifty move rule"
    if board.is_repetition():
        return "Draw - Threefold repetition"
    return "ongoing"


def get_legal_moves() -> str:
    moves = ",".join(move.uci() for move in board.legal_moves)
    debug_log(f"get_legal_moves returning: {moves}")
    log_board_snapshot("get_legal_moves snapshot")
    return moves


def log_event(msg: str) -> bool:
    print(f"[ARDUINO LOG]: {msg}")
    log_board_snapshot(f"log_event '{msg}'")
    return True


def camera_calibrate() -> str:
    global locked_outer_pts, last_stable_gray, square_occ_thresholds, board_rotation
    log_board_snapshot("camera_calibrate entry")
    debug_log(f"camera_calibrate called with board FEN: {board.fen()}")
    if not ensure_camera():
        set_camera_status("Calibration failed: no camera available.")
        return format_result("fail", camera_status)
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
            square_occ_thresholds = {}
            board_rotation = BOARD_ROTATION
            debug_log(f"Calibration corners locked on attempt {attempt + 1}: {locked_outer_pts.tolist()}")
            set_camera_status(f"Calibration successful on attempt {attempt + 1}.")
            return format_result("success", camera_status)
        time.sleep(0.2)
    set_camera_status("Calibration failed: board corners not found.")
    return format_result("fail", camera_status)


def camera_verify() -> str:
    log_board_snapshot("camera_verify entry")
    debug_log(f"camera_verify called with board FEN: {board.fen()}")
    if locked_outer_pts is None:
        set_camera_status("Verify failed: calibrate first.")
        return format_result("fail", camera_status)
    if not ensure_camera():
        set_camera_status("Verify failed: no camera available.")
        return format_result("fail", camera_status)
    success, frame = camera.read()
    if not success:
        set_camera_status("Verify failed: camera read failed.")
        return format_result("fail", camera_status)
    try:
        warped = apply_warp(frame, get_perspective_matrix())
        if warped is None or warped.size == 0:
            raise ValueError("empty warped image")
        debug_log(f"Verify produced warped image with shape {warped.shape}")
        set_camera_status("Verify successful: warped board image created.")
        return format_result("success", camera_status)
    except Exception as exc:
        set_camera_status(f"Verify failed: {exc}")
        return format_result("fail", camera_status)


def camera_capture_initial() -> str:
    global board, last_stable_gray, board_rotation
    log_board_snapshot("camera_capture_initial entry")
    debug_log(f"camera_capture_initial called before reset with board FEN: {board.fen()}")
    if locked_outer_pts is None:
        set_camera_status("Initial capture failed: calibrate first.")
        return format_result("fail", camera_status)
    if not ensure_camera():
        set_camera_status("Initial capture failed: no camera available.")
        return format_result("fail", camera_status)

    median_gray, last_bgr, stability_pct, gray_frames, last_raw = capture_stable_median(get_perspective_matrix())
    if stability_pct is None:
        set_camera_status("Initial capture failed: camera read failed.")
        return format_result("fail", camera_status)
    if median_gray is None:
        set_camera_status(f"Initial capture unstable ({stability_pct:.2f}%).")
        return format_result("fail", camera_status)

    if last_bgr is not None:
        chosen_name, detected_rotation, rotation_source = choose_starting_rotation(last_raw, last_bgr)
        board_rotation = detected_rotation
        median_gray = rotate_image(median_gray, detected_rotation)
        gray_frames = [rotate_image(gray, detected_rotation) for gray in gray_frames]
        debug_log(
            f"Applied startup board rotation {chosen_name} via {rotation_source}."
        )

    board = chess.Board()
    last_stable_gray = median_gray
    update_square_thresholds(gray_frames, median_gray)
    debug_log(f"Initial reference stored. mean={float(np.mean(median_gray)):.2f}, std={float(np.std(median_gray)):.2f}")
    debug_log(f"Initial capture reset board to FEN: {board.fen()}")
    set_camera_status("Initial board capture successful.")
    print(f"[PYTHON]: Board reset and initial reference stored: {board.fen()}")
    log_board_snapshot("camera_capture_initial exit")
    return format_result("success", camera_status)


def camera_capture_player_move() -> str:
    global last_stable_gray
    log_board_snapshot("camera_capture_player_move entry")
    debug_log(f"camera_capture_player_move called with expected board FEN: {board.fen()}")
    if locked_outer_pts is None:
        set_camera_status("Player capture failed: calibrate first.")
        return format_result("fail", camera_status)
    if last_stable_gray is None:
        set_camera_status("Player capture failed: initial capture missing.")
        return format_result("fail", camera_status)
    if not ensure_camera():
        set_camera_status("Player capture failed: no camera available.")
        return format_result("fail", camera_status)

    median_gray, _, stability_pct, _, _ = capture_stable_median(get_perspective_matrix())
    if stability_pct is None:
        set_camera_status("Player capture failed: camera read failed.")
        return format_result("fail", camera_status)
    if median_gray is None:
        set_camera_status(f"Player capture unstable ({stability_pct:.2f}%).")
        return format_result("fail", camera_status)

    diff_img = cv2.absdiff(median_gray, last_stable_gray)
    debug_log(
        f"Player capture diff stats: mean={float(np.mean(diff_img)):.2f}, max={int(np.max(diff_img))}, board_fen={board.fen()}"
    )
    move, _, changed_squares, _ = detect_move(diff_img)
    if move is None:
        changed_names = sorted(chess.square_name(s) for s in changed_squares)
        set_camera_status(f"Player move not detected. Changed: {changed_names}")
        debug_log(f"Player move detection failed while board expected FEN: {board.fen()}")
        log_board_snapshot("camera_capture_player_move fail")
        return format_result("fail", camera_status)

    debug_log(f"Player move candidate accepted: {move.uci()} on board FEN {board.fen()}")
    board.push(move)
    last_stable_gray = median_gray
    move_uci = move.uci()
    debug_log(f"Player move accepted: {move_uci}")
    debug_log(f"Board FEN after player push: {board.fen()}")
    set_camera_status(f"Player move detected: {move_uci}")
    print(f"[PYTHON]: Player move registered from camera -> {move_uci}")
    print(f"[PYTHON]: Board after player move: {board.fen()}")
    log_board_snapshot("camera_capture_player_move exit")
    return format_result("success", move_uci)


def camera_refresh_reference() -> str:
    global last_stable_gray
    log_board_snapshot("camera_refresh_reference entry")
    debug_log(f"camera_refresh_reference called with board FEN: {board.fen()}")
    if locked_outer_pts is None:
        set_camera_status("Reference refresh failed: calibrate first.")
        return format_result("fail", camera_status)
    if not ensure_camera():
        set_camera_status("Reference refresh failed: no camera available.")
        return format_result("fail", camera_status)

    attempt = 1
    while True:
        debug_log(f"camera_refresh_reference attempt {attempt}")
        median_gray, _, stability_pct, gray_frames, _ = capture_stable_median(get_perspective_matrix())
        if stability_pct is None:
            set_camera_status("Reference refresh failed: camera read failed.")
            return format_result("fail", camera_status)
        if median_gray is not None:
            break

        set_camera_status(f"Reference refresh unstable ({stability_pct:.2f}%). Retrying.")
        debug_log("Reference refresh rejected as unstable, retrying.")
        attempt += 1
        time.sleep(0.2)

    last_stable_gray = median_gray
    update_square_thresholds(gray_frames, median_gray)
    debug_log(f"Reference refreshed. mean={float(np.mean(median_gray)):.2f}, std={float(np.std(median_gray)):.2f}")
    debug_log(f"Reference refresh kept board FEN as: {board.fen()}")
    set_camera_status("Reference refreshed after robot move.")
    log_board_snapshot("camera_refresh_reference exit")
    return format_result("success", camera_status)


def get_camera_status() -> str:
    debug_log(f"get_camera_status returning '{camera_status}' with board FEN {board.fen()}")
    log_board_snapshot("get_camera_status snapshot")
    return camera_status


Bridge.provide("log_event", log_event)
Bridge.provide("get_move", get_robot_move)
Bridge.provide("reset_game", reset_game)
Bridge.provide("get_board_state", get_board_state)
Bridge.provide("is_game_over", is_game_over)
Bridge.provide("get_legal_moves", get_legal_moves)
Bridge.provide("camera_calibrate", camera_calibrate)
Bridge.provide("camera_verify", camera_verify)
Bridge.provide("camera_capture_initial", camera_capture_initial)
Bridge.provide("camera_capture_player_move", camera_capture_player_move)
Bridge.provide("camera_refresh_reference", camera_refresh_reference)
Bridge.provide("get_camera_status", get_camera_status)

if initialize_engine():
    print("[PYTHON]: Chess and vision bridge ready")
    print(f"[PYTHON]: Starting FEN: {board.fen()}")
else:
    print("[PYTHON ERROR]: System started without a working engine")

App.run()
