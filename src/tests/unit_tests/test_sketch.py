import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SKETCH_PATH = ROOT / "src" / "sketch.ino"


class TestSketchProtocolSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sketch_source = SKETCH_PATH.read_text(encoding="utf-8", errors="replace")

    def test_sketch_contains_expected_bridge_commands(self):
        for command in (
            "camera_calibrate",
            "camera_verify",
            "camera_capture_initial",
            "camera_capture_player_move",
            "camera_refresh_reference",
            "get_move",
            "reset_game",
        ):
            self.assertIn(command, self.sketch_source)

    def test_setup_homes_once_and_button_camera_homing_is_disabled(self):
        self.assertIn("initializePosition();\n  rotateX90(false);", self.sketch_source)
        self.assertIn("//retractArmToHome();\n    bool success = callBridgeStep(\"camera_calibrate\", data);", self.sketch_source)
        self.assertIn("//retractArmToHome();\n    bool success = callBridgeStep(\"camera_verify\", data);", self.sketch_source)
        self.assertIn("//retractArmToHome();\n    bool success = callBridgeStep(\"camera_capture_initial\", data);", self.sketch_source)
        self.assertIn("//retractArmToHome();\n    String data;\n    bool success = callBridgeStep(\"camera_capture_player_move\", data);", self.sketch_source)

    def test_sketch_uses_single_button_game_flow(self):
        self.assertIn("BridgeClass Bridge(Serial1);", self.sketch_source)
        self.assertIn("#define BUTTON_OK A0", self.sketch_source)
        self.assertIn("if (buttonOkPressed()) {", self.sketch_source)
        self.assertIn("buttonOkLongPress", self.sketch_source)
        self.assertIn("Game resigned.", self.sketch_source)

    def test_button_release_guard_is_present(self):
        self.assertIn("bool buttonOkPressedLatch = false;", self.sketch_source)
        self.assertIn("bool buttonOkReleaseRequired = false;", self.sketch_source)
        self.assertIn("void requireButtonRelease()", self.sketch_source)
        self.assertIn("if (!buttonOkReleaseRequired && buttonOkState", self.sketch_source)
        self.assertIn("requireButtonRelease();\n  Bridge.call(\"log_event\", \"Game started. Robot is White.\");", self.sketch_source)

    def test_sketch_has_square_parsing_validation(self):
        self.assertIn("if (square == nullptr || strlen(square) < 2) return false;", self.sketch_source)
        self.assertIn("if (file < 'A' || file > 'H' || rank < 1 || rank > 8) return false;", self.sketch_source)

    def test_sketch_retract_helper_uses_absolute_homing_steps(self):
        self.assertIn("void retractArmToHome()", self.sketch_source)
        self.assertIn("initializePosition();", self.sketch_source)
        self.assertIn("rotateX90(false);", self.sketch_source)
        self.assertIn("setGripper(GRIPPER_CLOSED);", self.sketch_source)

    def test_execute_move_handles_capture_before_source_pickup(self):
        capture_index = self.sketch_source.index("if (p2 != 0) {")
        pickup_index = self.sketch_source.index("if (!pickupFromSquare(sq1)) {")
        place_index = self.sketch_source.index("if (!placeToSquare(sq2)) {")
        self.assertLess(capture_index, pickup_index)
        self.assertLess(pickup_index, place_index)

    def test_request_robot_move_parses_bridge_payload_and_refreshes_reference(self):
        self.assertIn("sscanf(data.c_str(), \"%[^,],%[^,],%d,%d\", s1, s2, &p1, &p2) == 4", self.sketch_source)
        self.assertIn("executeUCIMove(s1, s2, p1, p2);", self.sketch_source)
        self.assertIn("callBridgeStep(\"camera_refresh_reference\", refreshData);", self.sketch_source)
        self.assertIn("toggleTurn();", self.sketch_source)

    def test_start_game_sets_expected_core_state(self):
        self.assertIn("gameActive = true;", self.sketch_source)
        self.assertIn("setupComplete = true;", self.sketch_source)
        self.assertIn("robotTurn = true;", self.sketch_source)
        self.assertIn("robotNextMove = \"----\";", self.sketch_source)

    def test_player_turn_only_captures_when_player_is_active(self):
        self.assertIn("if (buttonOkPressed() && gameActive && !robotTurn)", self.sketch_source)
        self.assertIn("callBridgeStep(\"camera_capture_player_move\", data);", self.sketch_source)

    def test_loop_only_requests_robot_move_on_robot_turn(self):
        self.assertIn("if (gameActive && robotTurn) {", self.sketch_source)
        self.assertIn("requestAndExecuteRobotMove();", self.sketch_source)

    def test_bridge_step_only_treats_success_prefix_as_success(self):
        self.assertIn("return data.startsWith(\"success:\");", self.sketch_source)


if __name__ == "__main__":
    unittest.main()
