import cv2
import chess
import chess.engine
import numpy as np
import time
from collections import deque
import os

"""
METHODS:

__init__(): Initialize camera(640x480@10fps), chess.Board(), Stockfish engine, motion tracking

_detect_motion(gray): Returns True if frame stable (<1.5% pixel change vs prev)

_wait_stable(timeout=30): Waits for stable_count>=5 frames, returns gray frame

_find_board_corners(gray): Hough lines → intersections → convex hull → 4 ordered corners (TL/TR/BR/BL)

_line_intersect(l1,l2): Computes 2D line intersection or None

_order_corners(pts): Sorts 4pts clockwise from top-left (sum/diff heuristics)  

_get_square_occupancy(gray): Perspective warp → 8x8 grid → adaptive thresh → contour area>12%=occupied

_infer_move(old_occ,new_occ): Finds changed squares → matches legal_moves (handles castling/en passant)

detect_human_move(): Full cycle: baseline → wait motion → post-move → infer → push to board

analyze_and_move(): Stockfish 1s analysis → returns best PV[0] (random fallback)

execute_physical_move(move): TODO: pyFirmata Arduino bridge - home→pickup→move→drop→home

close(): Cleanup camera/engine resources
"""

class ChessRobot:
    def __init__(self, stockfish_path=None):
        self.board = chess.Board()
        self.cap = cv2.VideoCapture(0)
        # not needed to have any higher res
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path or "stockfish") if stockfish_path else None
        self.motion_history = deque(maxlen=10) # not needed to have more than 10, we still need to save memory
        self.stable_threshold = 5
        self.last_occupancy = None
        
    def _detect_motion(self, gray):
        """Returns True if LOW motion (stable)"""
        if len(self.motion_history) < 2:
            self.motion_history.append(gray.copy())
            return False
        
        prev = self.motion_history[-1]
        diff = cv2.absdiff(gray, prev)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_ratio = np.sum(thresh > 0) / (gray.shape[0] * gray.shape[1])
        
        self.motion_history.append(gray.copy())
        return motion_ratio < 0.015  # stable if <1.5% changem can be and should be changed prob
    
    def _wait_stable(self, timeout=30):
        """Wait for stable board, return gray or None"""
        start = time.time()
        stable_count = 0
        # print("Waiting for stable board...")
        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._detect_motion(gray):
                stable_count += 1
                if stable_count >= self.stable_threshold:
                    print("Stable!")
                    return gray
            else:
                stable_count = 0
            time.sleep(0.1)
        print("Timeout waiting for stable")
        return None
    
    def _find_board_corners(self, gray): # AI pretty much made the entire hough line interpret 
        """Dynamic corner detection via Hough lines for physical board [web:30][web:40]"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)
        if lines is None:
            return None
        
        # separate horiz&vert lines
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle - 180) < 10:  # horiz
                h_lines.append(line[0])
            elif 80 < abs(angle) < 100:  # vert
                v_lines.append(line[0])
        
        if len(h_lines) < 4 or len(v_lines) < 4:
            return None
        
        # find intersections (approx 4 outer corners)
        corners = []
        for h in h_lines[:8]:  # top few longest
            for v in v_lines[:8]:
                pt = self._line_intersect(h, v)
                if pt:
                    corners.append(pt)
        
        if len(corners) < 4:
            return None
        
        # Largest quadrilateral (convex hull)
        hull = cv2.convexHull(np.array(corners), clockwise=False)
        if len(hull) >= 4:
            pts = hull.reshape(-1, 2).astype(np.float32)
            # Order: TL, TR, BR, BL
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            box = self._order_corners(box)
            return box
        
        return None
    
    def _line_intersect(self, l1, l2):
        """Intersection of two lines"""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))
    
    def _order_corners(self, pts):
        """Order corners clockwise from TL"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect
    
    def _get_square_occupancy(self, gray): # AI highly improved my function here
        """Occupancy for wooden black/white pieces [web:22][web:44]"""
        corners = self._find_board_corners(gray)
        if corners is None:
            return {}
        
        dst_size = 400
        dst_pts = np.float32([[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]])
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(gray, matrix, (dst_size, dst_size))
        
        occupancy = {}
        square_size = dst_size // 8
        
        for rank in range(8):
            for file in range(8):
                y, x = rank * square_size, file * square_size
                sq = warped[y:y+square_size, x:x+square_size]
                
                # adaptive thresh + contour algo for wood texture
                blur = cv2.GaussianBlur(sq, (7, 7), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                area_ratio = 0
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area_ratio = cv2.contourArea(largest) / (square_size ** 2)
                
                sq_idx = chess.square(file, 7 - rank)  # a8=0, h1=63
                occupancy[sq_idx] = area_ratio > 0.12  # tune this for the actual board!!

        cv2.imshow('warped', warped)
        cv2.imshow('thresh_example', thresh)  # Last square
        if cv2.waitKey(1) & 0xFF == ord('q'): exit()
        return occupancy
    
    def _infer_move(self, old_occ, new_occ):
        """Same as original, improved"""
        changed = [sq for sq in range(64) if old_occ.get(sq, False) != new_occ.get(sq, False)]
        if len(changed) < 2:
            return None
        
        for move in list(self.board.legal_moves):
            expected = {move.from_square, move.to_square}
            if self.board.is_castling(move):
                if move.to_square > move.from_square:
                    expected |= {move.from_square + 1, move.from_square + 2}  # rook lands
                else:
                    expected |= {move.from_square - 1, move.from_square - 2}
            elif self.board.is_en_passant(move):
                dir = -8 if self.board.turn == chess.WHITE else 8
                expected.add(move.to_square + dir)  # captured pawn
            
            if set(changed) == expected:
                return move
        return None
    
    def detect_human_move(self):
        """Full cycle"""
        #print("\n Detecting human move ")
        
        gray1 = self._wait_stable()
        if gray1 is None:
            return None
        old_occ = self._get_square_occupancy(gray1)
        self.last_occupancy = old_occ
        
        # wait for motion (human started move)
        print("Make your move...")
        start = time.time()
        while time.time() - start < 60:  # 1min wait for humans move could be a lot less 
            ret, frame = self.cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self._detect_motion(gray):  # motion detected!
                break
            time.sleep(0.1)
        else:
            print("Move timeout")
            return None
        
        # wait till stable again
        gray2 = self._wait_stable()
        if gray2 is None:
            return None
        new_occ = self._get_square_occupancy(gray2)
        
        move = self._infer_move(old_occ, new_occ)
        if move:
            print(f"Detected: {move}")
            self.board.push(move)
            return move
        
        print("No valid move detected")
        return None
    
    def analyze_and_move(self):
        """stockfish move"""
        if self.engine is None:
            print("No Stockfish - random legal move")
            return self.board.legal_moves.__iter__().__next__()
        
        result = self.engine.analyse(self.board, chess.engine.Limit(time=1.0))
        move = result['pv'][0]
        print(f"Stockfish: {move}")
        return move
    
    def close(self):
        self.cap.release()
        if self.engine:
            self.engine.quit()

    def execute_physical_move(self, move):
        """Done with bridging to C"""
        # possibilities: pyFirmata, Jhonny5 ecosystem, pyserial
        
        """"
        Should be something like this but with bridging

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        
        self.robot_arm.extend_to_square(from_sq)  # move above from
        self.gripper.pickup()  # grab
        self.robot_arm.move_to_square(to_sq)      # move to 
        self.gripper.release()                    # drop
        self.robot_arm.retract_to_base()
        """
def main():
    # stockfish https://stockfishchess.org/download/
    try:
        robot = ChessRobot(stockfish_path="stockfish")  # or w.e path
    except ImportError as e:
        print("Import error occured:", e)
    # print("ready")
    # input()
    
    try:
        while not robot.board.is_game_over():
            # human move
            human_move = robot.detect_human_move()
            if not human_move:
                continue
            robot.board.push(human_move)
            
            # robot moves
            robot_move = robot.analyze_and_move()
            robot.board.push(robot_move)
            print("Robot move executed (add servo code here)")
            #time.sleep(2) # imagine we actually moved
            
        print(f"Game over: {robot.board.result()}")
    finally:
        robot.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
