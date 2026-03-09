import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import chess
import cv2
from main import ChessRobot

# In order to operat unittesting please ensure you execute the following line, unless already downloaded
# pip install numpy opencv-python python-chess


class TestChessRobot(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def setUp(self, mock_video_capture):
        # Mock the camera so it doesn't actually try to open
        self.mock_cap = MagicMock()
        mock_video_capture.return_value = self.mock_cap
        
        # Initialize robot without a real stockfish path for basic logic tests
        self.robot = ChessRobot(stockfish_path=None)

    def test_detect_motion_stable(self):
        """Test that detect_motion returns True when frames are identical."""
        # Create two identical "frames" (black images)
        frame = np.zeros((480, 640), dtype=np.uint8)
        
        # First call populates history
        self.robot._detect_motion(frame)
        # Second call compares against the first
        is_stable = self.robot._detect_motion(frame)
        
        self.assertTrue(is_stable, "Identical frames should be detected as stable.")

    def test_detect_motion_movement(self):
        """Test that detect_motion returns False when frames are very different."""
        frame1 = np.zeros((480, 640), dtype=np.uint8)
        frame2 = np.ones((480, 640), dtype=np.uint8) * 255 # Pure white
        
        self.robot._detect_motion(frame1)
        is_stable = self.robot._detect_motion(frame2)
        
        self.assertFalse(is_stable, "Significantly different frames should not be stable.")

    def test_infer_move_standard(self):
        """Test move inference for a simple pawn move e2e4."""
        # Setup board state
        self.robot.board = chess.Board() # Starting position
        
        # e2 is square 12, e4 is square 28
        old_occ = {i: True for i in range(16)} # Simplify: ranks 1 & 2 occupied
        new_occ = old_occ.copy()
        new_occ[12] = False # e2 empty
        new_occ[28] = True  # e4 occupied
        
        move = self.robot._infer_move(old_occ, new_occ)
        
        self.assertIsNotNone(move)
        self.assertEqual(move.uci(), "e2e4")

    def test_order_corners(self):
        """Test the clockwise sorting logic for board corners."""
        # Unordered points: [BL, TR, BR, TL]
        pts = np.array([[0, 100], [100, 0], [100, 100], [0, 0]], dtype=np.float32)
        ordered = self.robot._order_corners(pts)
        
        # Expected: TL(0,0), TR(100,0), BR(100,100), BL(0,100)
        np.testing.assert_array_equal(ordered[0], [0, 0])
        np.testing.assert_array_equal(ordered[2], [100, 100])


    def test_board_detection_with_real_image(self):
        """Verify corner detection works on a real static image."""
        # 1. Load a real image from your folder
        image_path = 'board_test.jpg'
        
        if not os.path.exists(image_path):
            self.skipTest(f"Skipping: {image_path} not found. Place a photo of your board in the folder.")

        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Run the detection logic
        corners = self.robot._find_board_corners(gray)

        # 3. Assertions
        self.assertIsNotNone(corners, "Failed to detect board corners in the test image.")
        self.assertEqual(len(corners), 4, "Should have found exactly 4 corners.")
        
        # Check that the corners form a reasonable area (not just a tiny speck)
        # TL to BR distance should be significant
        dist = np.linalg.norm(corners[0] - corners[2])
        self.assertGreater(dist, 100, "Detected board area is suspiciously small.")

    def tearDown(self):
        self.robot.close()

if __name__ == '__main__':
    unittest.main()