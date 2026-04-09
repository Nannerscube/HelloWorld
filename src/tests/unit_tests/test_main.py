import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

try:
    import chess
    import cv2
    import numpy as np
    DEPS_AVAILABLE = True
except ModuleNotFoundError:
    chess = None
    cv2 = None
    np = None
    DEPS_AVAILABLE = False


ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
MAIN_PATH = SRC_DIR / "main.py"


class DummyBridge:
    registered = {}

    @classmethod
    def reset(cls):
        cls.registered = {}

    @classmethod
    def provide(cls, name, func):
        cls.registered[name] = func

    @classmethod
    def call(cls, *args, **kwargs):
        raise NotImplementedError("Bridge.call is not used directly in these unit tests.")


class DummyApp:
    run_called = False

    @classmethod
    def reset(cls):
        cls.run_called = False

    @staticmethod
    def run():
        DummyApp.run_called = True


class FakeEngine:
    def __init__(self, move_uci="e2e4"):
        self.move_uci = move_uci
        self.configure_calls = []
        self.quit_called = False
        self.play_calls = []

    def configure(self, options):
        self.configure_calls.append(options)

    def play(self, board, limit):
        self.play_calls.append((board.fen(), limit))
        return types.SimpleNamespace(move=chess.Move.from_uci(self.move_uci))

    def quit(self):
        self.quit_called = True


def load_main_module():
    if not DEPS_AVAILABLE:
        raise unittest.SkipTest("python-chess/opencv-python/numpy are required for these tests.")

    DummyBridge.reset()
    DummyApp.reset()

    fake_app_utils = types.ModuleType("arduino.app_utils")
    fake_app_utils.App = DummyApp
    fake_app_utils.Bridge = DummyBridge
    fake_arduino = types.ModuleType("arduino")
    fake_arduino.app_utils = fake_app_utils

    sys.modules["arduino"] = fake_arduino
    sys.modules["arduino.app_utils"] = fake_app_utils
    sys.modules.pop("tested_main", None)

    fake_engine = FakeEngine()

    with patch("chess.engine.SimpleEngine.popen_uci", return_value=fake_engine):
        spec = importlib.util.spec_from_file_location("tested_main", MAIN_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return module, fake_engine


@unittest.skipUnless(DEPS_AVAILABLE, "python-chess/opencv-python/numpy are required")
class TestMainPy(unittest.TestCase):
    def setUp(self):
        self.main, self.fake_engine = load_main_module()

    def tearDown(self):
        self.main.close_engine()
        self.main.close_camera()
        sys.modules.pop("tested_main", None)
        sys.modules.pop("arduino", None)
        sys.modules.pop("arduino.app_utils", None)

    def test_bridge_functions_are_registered(self):
        expected = {
            "log_event",
            "get_move",
            "reset_game",
            "get_board_state",
            "is_game_over",
            "get_legal_moves",
            "camera_calibrate",
            "camera_verify",
            "camera_capture_initial",
            "camera_capture_player_move",
            "camera_refresh_reference",
            "get_camera_status",
        }
        self.assertTrue(DummyApp.run_called)
        self.assertEqual(expected, set(DummyBridge.registered))

    def test_initialize_engine_configures_skill_level(self):
        self.assertIs(self.main.engine, self.fake_engine)
        self.assertIn({"Skill Level": self.main.SKILL_LEVEL}, self.fake_engine.configure_calls)

    def test_format_result(self):
        self.assertEqual(self.main.format_result("success", "ok"), "success:ok")

    def test_rotation_name_and_bottom_side_mapping(self):
        self.assertEqual(self.main.rotation_name(cv2.ROTATE_180), "rotate_180")
        self.assertEqual(self.main.rotation_name(None), "none")
        self.assertEqual(self.main.get_rotation_for_bottom_side("left"), cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.assertEqual(self.main.get_rotation_for_bottom_side("bottom"), None)

    def test_uci_to_piece_type_maps_starting_pieces(self):
        self.main.board = chess.Board()
        self.assertEqual(self.main.uci_to_piece_type("e2"), 1)
        self.assertEqual(self.main.uci_to_piece_type("b1"), 3)
        self.assertEqual(self.main.uci_to_piece_type("c1"), 4)
        self.assertEqual(self.main.uci_to_piece_type("a1"), 5)
        self.assertEqual(self.main.uci_to_piece_type("d1"), 6)
        self.assertEqual(self.main.uci_to_piece_type("e1"), 2)
        self.assertEqual(self.main.uci_to_piece_type("e4"), 0)

    def test_get_move_footprint_adds_castling_rook_squares(self):
        self.main.board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1g1")
        footprint = self.main.get_move_footprint(move)
        names = {chess.square_name(square) for square in footprint}
        self.assertEqual(names, {"e1", "g1", "h1", "f1"})

    def test_get_move_footprint_adds_en_passant_capture_square(self):
        self.main.board = chess.Board("8/8/8/3pP3/8/8/8/8 w - d6 0 1")
        move = chess.Move.from_uci("e5d6")
        footprint = self.main.get_move_footprint(move)
        names = {chess.square_name(square) for square in footprint}
        self.assertEqual(names, {"e5", "d6", "d5"})

    def test_reset_game_restores_starting_board(self):
        self.main.board.push(chess.Move.from_uci("e2e4"))
        self.assertTrue(self.main.reset_game())
        self.assertEqual(self.main.board.fen(), chess.Board().fen())

    def test_get_board_state_and_legal_moves(self):
        board = chess.Board()
        self.main.board = board
        self.assertEqual(self.main.get_board_state(), board.fen())
        legal_moves = self.main.get_legal_moves().split(",")
        self.assertIn("e2e4", legal_moves)
        self.assertIn("g1f3", legal_moves)

    def test_is_game_over_variants(self):
        self.main.board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(self.main.is_game_over(), "Checkmate! Player wins!")

        self.main.board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
        self.assertEqual(self.main.is_game_over(), "Stalemate - Draw!")

    def test_log_event_returns_true(self):
        self.assertTrue(self.main.log_event("hello"))

    def test_get_robot_move_uses_engine_and_pushes_board(self):
        self.main.board = chess.Board()
        move_str = self.main.get_robot_move()
        self.assertEqual(move_str, "e2,e4,1,0")
        self.assertEqual(self.main.board.peek().uci(), "e2e4")
        self.assertEqual(self.main.board.turn, chess.BLACK)

    def test_get_robot_move_falls_back_when_engine_missing(self):
        self.main.engine = None
        self.main.board = chess.Board()
        self.assertEqual(self.main.get_robot_move(), "e2,e4,1,0")
        self.assertEqual(self.main.board.fen(), chess.Board().fen())

    def test_camera_verify_requires_calibration(self):
        self.main.locked_outer_pts = None
        result = self.main.camera_verify()
        self.assertEqual(result, "fail:Verify failed: calibrate first.")

    def test_camera_capture_initial_requires_calibration(self):
        self.main.locked_outer_pts = None
        result = self.main.camera_capture_initial()
        self.assertEqual(result, "fail:Initial capture failed: calibrate first.")

    def test_camera_capture_player_move_requires_initial_reference(self):
        self.main.locked_outer_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.main.last_stable_gray = None
        result = self.main.camera_capture_player_move()
        self.assertEqual(result, "fail:Player capture failed: initial capture missing.")

    def test_camera_refresh_reference_requires_calibration(self):
        self.main.locked_outer_pts = None
        result = self.main.camera_refresh_reference()
        self.assertEqual(result, "fail:Reference refresh failed: calibrate first.")

    def test_camera_calibrate_fails_without_camera(self):
        self.main.camera = None
        with patch.object(self.main, "ensure_camera", return_value=False):
            result = self.main.camera_calibrate()
        self.assertEqual(result, "fail:Calibration failed: no camera available.")

    def test_camera_verify_success_path(self):
        self.main.locked_outer_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        self.main.camera = MagicMock()
        self.main.camera.read.return_value = (True, np.zeros((20, 20, 3), dtype=np.uint8))
        with patch.object(self.main, "ensure_camera", return_value=True), \
             patch.object(self.main, "get_perspective_matrix", return_value=np.eye(3, dtype=np.float32)):
            result = self.main.camera_verify()
        self.assertEqual(result, "success:Verify successful: warped board image created.")

    def test_camera_capture_initial_success_resets_board_and_reference(self):
        self.main.locked_outer_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        self.main.board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        median = np.full((self.main.BOARD_SIZE, self.main.BOARD_SIZE), 120, dtype=np.uint8)
        gray_frames = [median.copy(), median.copy(), median.copy()]

        with patch.object(self.main, "ensure_camera", return_value=True), \
             patch.object(
                 self.main,
                 "capture_stable_median",
                 return_value=(
                     median,
                     np.zeros((self.main.BOARD_SIZE, self.main.BOARD_SIZE, 3), dtype=np.uint8),
                     0.5,
                     gray_frames,
                     np.zeros((20, 20, 3), dtype=np.uint8),
                 ),
             ), \
             patch.object(self.main, "get_perspective_matrix", return_value=np.eye(3, dtype=np.float32)), \
             patch.object(self.main, "choose_starting_rotation", return_value=("none", None, "board-heuristic")):
            result = self.main.camera_capture_initial()

        self.assertEqual(result, "success:Initial board capture successful.")
        self.assertEqual(self.main.board.fen(), chess.Board().fen())
        self.assertIsNotNone(self.main.last_stable_gray)
        self.assertEqual(self.main.board_rotation, None)

    def test_camera_capture_player_move_success_pushes_board(self):
        self.main.locked_outer_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        self.main.last_stable_gray = np.zeros((self.main.BOARD_SIZE, self.main.BOARD_SIZE), dtype=np.uint8)
        self.main.board = chess.Board()

        with patch.object(self.main, "ensure_camera", return_value=True), \
             patch.object(
                 self.main,
                 "capture_stable_median",
                 return_value=(np.ones((self.main.BOARD_SIZE, self.main.BOARD_SIZE), dtype=np.uint8), None, 0.5, None, None),
             ), \
             patch.object(
                 self.main,
                 "detect_move",
                 return_value=(chess.Move.from_uci("e2e4"), "e4", {chess.E2, chess.E4}, {"e2": 0.5, "e4": 0.6}),
             ), \
             patch.object(self.main, "get_perspective_matrix", return_value=np.eye(3, dtype=np.float32)):
            result = self.main.camera_capture_player_move()

        self.assertEqual(result, "success:e2e4")
        self.assertEqual(self.main.board.peek().uci(), "e2e4")

    def test_camera_refresh_reference_success_updates_reference(self):
        self.main.locked_outer_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        median = np.full((self.main.BOARD_SIZE, self.main.BOARD_SIZE), 100, dtype=np.uint8)
        gray_frames = [median.copy(), median.copy(), median.copy()]

        with patch.object(self.main, "ensure_camera", return_value=True), \
             patch.object(
                 self.main,
                 "capture_stable_median",
                 return_value=(median, None, 0.3, gray_frames, None),
             ), \
             patch.object(self.main, "get_perspective_matrix", return_value=np.eye(3, dtype=np.float32)):
            result = self.main.camera_refresh_reference()

        self.assertEqual(result, "success:Reference refreshed after robot move.")
        self.assertIs(self.main.last_stable_gray, median)


if __name__ == "__main__":
    unittest.main()
