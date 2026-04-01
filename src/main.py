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
STABILITY_MAX_PCT = 2.5
MIN_FOOTPRINT_IOU = 0.5
MAX_CHANGED_SQUARES = 6
BOARD_ROTATION = None

board = chess.Board()
engine = None
camera = None
locked_outer_pts = None
last_stable_gray = None
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
    for candidate in ENGINE_PATH_CANDIDATES:
        if not candidate:
            continue
        try:
            engine = chess.engine.SimpleEngine.popen_uci(candidate)
            engine.configure({"Skill Level": SKILL_LEVEL})
            print(f"[PYTHON]: Stockfish ready from {candidate}")
            debug_log(f"Engine initialized with starting board FEN: {board.fen()}")
            return True
        except Exception:
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


def get_square_center_roi(gray_img, row, col, crop_ratio=0.45):
    square_size = BOARD_SIZE // 8
    inset = int(square_size * (1 - crop_ratio) / 2)
    y1 = row * square_size + inset
    y2 = (row + 1) * square_size - inset
    x1 = col * square_size + inset
    x2 = (col + 1) * square_size - inset
    return gray_img[y1:y2, x1:x2]


def detect_starting_orientation(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    top_samples = []
    bottom_samples = []

    for row in [0, 1]:
        for col in range(8):
            roi = get_square_center_roi(gray, row, col)
            if roi.size:
                top_samples.append(float(np.mean(roi)))

    for row in [6, 7]:
        for col in range(8):
            roi = get_square_center_roi(gray, row, col)
            if roi.size:
                bottom_samples.append(float(np.mean(roi)))

    top_mean = float(np.mean(top_samples)) if top_samples else 0.0
    bottom_mean = float(np.mean(bottom_samples)) if bottom_samples else 0.0
    should_flip = top_mean > bottom_mean

    debug_log(
        f"Orientation probe: top_mean={top_mean:.2f}, bottom_mean={bottom_mean:.2f}, should_flip={should_flip}"
    )
    return should_flip, top_mean, bottom_mean


def capture_stable_median(matrix):
    debug_log("Starting stable median capture.")
    time.sleep(0.5)
    for _ in range(2):
        camera.read()

    grays = []
    last_bgr = None
    start = time.time()
    for i in range(3):
        wait = (start + i * FRAME_DELAY) - time.time()
        if wait > 0:
            time.sleep(wait)
        success, frame = camera.read()
        if not success:
            debug_log("Camera read failed during stable median capture.")
            return None, None, None
        warped = apply_warp(frame, matrix)
        grays.append(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
        last_bgr = warped

    stack = np.stack([g.astype(np.float32) for g in grays], axis=0)
    stability_pct = float(round(np.mean(np.ptp(stack, axis=0)) / 255.0 * 100, 2))
    debug_log(f"Stable median capture finished with stability={stability_pct:.2f}%")
    if stability_pct > STABILITY_MAX_PCT:
        debug_log("Stable median capture rejected as unstable.")
        return None, last_bgr, stability_pct

    median_gray = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    debug_log("Stable median capture accepted.")
    return median_gray, last_bgr, stability_pct


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


def square_changed(diff_img, row, col):
    y1, y2, x1, x2 = get_square_roi_bounds(row, col)
    roi = diff_img[y1:y2, x1:x2]
    area = (y2 - y1) * (x2 - x1)
    if area == 0:
        return False, 0.0

    kernel = np.ones((3, 3), np.uint8)
    otsu_val, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val < OTSU_MIN_THRESH:
        _, thresh = cv2.threshold(roi, FIXED_FALLBACK_THRESH, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    ratio = cv2.countNonZero(thresh) / area
    return ratio > OCC_RATIO_THRESH, round(ratio, 3)


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
    if len(changed_squares) > MAX_CHANGED_SQUARES:
        names = sorted(chess.square_name(s) for s in changed_squares)
        debug_log(f"Too many changed squares detected: {names}")
        return None, f"Too many squares changed: {names}", changed_squares, ratios

    best_iou = -1.0
    best_moves = []
    for move in board.legal_moves:
        footprint = get_move_footprint(move)
        intersection = len(footprint & changed_squares)
        union = len(footprint | changed_squares)
        iou = intersection / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_moves = [move]
        elif iou == best_iou:
            best_moves.append(move)

    if not best_moves:
        debug_log("No legal moves available while trying to detect move.")
        return None, "No legal moves available.", changed_squares, ratios

    best_move = next((m for m in best_moves if m.promotion == chess.QUEEN), best_moves[0])
    changed_names = sorted(chess.square_name(s) for s in changed_squares)
    best_footprint = sorted(chess.square_name(s) for s in get_move_footprint(best_move))
    debug_log(f"Detected changed squares: {changed_names}")
    debug_log(f"Best move candidate: {best_move.uci()} with footprint {best_footprint} and IoU {best_iou:.2f}")
    if best_iou < MIN_FOOTPRINT_IOU:
        return None, f"No legal move match for {changed_names} (IoU {best_iou:.0%})", changed_squares, ratios
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
    global locked_outer_pts, last_stable_gray, board_rotation
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

    median_gray, last_bgr, stability_pct = capture_stable_median(get_perspective_matrix())
    if stability_pct is None:
        set_camera_status("Initial capture failed: camera read failed.")
        return format_result("fail", camera_status)
    if median_gray is None:
        set_camera_status(f"Initial capture unstable ({stability_pct:.2f}%).")
        return format_result("fail", camera_status)

    if last_bgr is not None:
        should_flip, top_mean, bottom_mean = detect_starting_orientation(last_bgr)
        if should_flip:
            board_rotation = cv2.ROTATE_180
            median_gray = cv2.rotate(median_gray, cv2.ROTATE_180)
            debug_log(
                f"Applied 180-degree board rotation after initial capture. top_mean={top_mean:.2f}, bottom_mean={bottom_mean:.2f}"
            )
        else:
            board_rotation = None
            debug_log(
                f"Kept board rotation unchanged after initial capture. top_mean={top_mean:.2f}, bottom_mean={bottom_mean:.2f}"
            )

    board = chess.Board()
    last_stable_gray = median_gray
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

    median_gray, _, stability_pct = capture_stable_median(get_perspective_matrix())
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
        median_gray, _, stability_pct = capture_stable_median(get_perspective_matrix())
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
