import cv2
import numpy as np
import chess
import sys

class ChessBoardRecognizer:
    def __init__(self):
        pass  # Reuse methods from ChessRobot

    def recognize_from_image(self, image_path):
        # Load single image instead of camera
        frame = cv2.imread(image_path)
        self.frame = frame
        if frame is None:
            print(f"Error loading image: {image_path}")
            return None, None, {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Reuse _find_board_corners
        corners = self._find_board_corners(gray)
        if corners is None:
            print("No board corners detected")
            return None, frame, {} # Return original for debug
        
        # Visualize detected board
        board_img = cv2.polylines(frame.copy(), [corners.astype(np.int32)], True, (0,255,0), 3)
        
        # Reuse _get_square_occupancy (it calls _find_board_corners internally, but we can pass gray)
        occupancy = self._get_square_occupancy(gray)
        
        # Convert occupancy to FEN (simplified - assumes occupancy only, pieces inferred if you have board state)
        fen = self.occupancy_to_fen(occupancy)
        
        return board_img, fen, occupancy

    def _find_board_corners(self, gray):
        # Copy-paste from ChessRobot._find_board_corners
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)
        if lines is None:
            return None
        
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle - 180) < 10:
                h_lines.append(line[0])
            elif 80 < abs(angle) < 100:
                v_lines.append(line[0])
        
        if len(h_lines) < 4 or len(v_lines) < 4:
            return None
        
        corners = []
        for h in h_lines[:8]:
            for v in v_lines[:8]:
                pt = self._line_intersect(h, v)
                if pt:
                    corners.append(pt)
        
        if len(corners) < 4:
            return None
        
        hull = cv2.convexHull(np.array(corners), clockwise=False)
        if len(hull) >= 4:
            pts = hull.reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            box = self._order_corners(box)
            return box
        return None

    def _line_intersect(self, l1, l2):
        # Copy from ChessRobot
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))

    def _order_corners(self, pts):
        # Copy from ChessRobot
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def _get_square_occupancy(self, gray):
        corners = self._find_board_corners(gray)
        if corners is None:
            return {}

        dst_size = 480
        dst_pts = np.float32([[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]])
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(gray, matrix, (dst_size, dst_size))

        trim = int(dst_size * 0.04)
        inner = warped[trim:dst_size - trim, trim:dst_size - trim]
        inner_size = inner.shape[0]
        square_size = inner_size // 8

        # ✅ Compute edges once for the whole board
        edges = cv2.Canny(inner, 30, 90)

        occupancy = {}
        brightness = np.zeros((8, 8))
        std_map = np.zeros((8, 8))
        edge_map = np.zeros((8, 8))

        for rank in range(8):
            for file in range(8):
                x = file * square_size
                y = rank * square_size
                margin = square_size // 4
                patch       = inner[y+margin:y+square_size-margin, x+margin:x+square_size-margin]
                edge_patch  = edges[y+margin:y+square_size-margin, x+margin:x+square_size-margin]

                brightness[rank, file] = np.mean(patch)
                std_map[rank, file]    = np.std(patch)
                # Edge density: fraction of edge pixels in the center patch
                edge_map[rank, file]   = np.sum(edge_patch > 0) / edge_patch.size

        light_mean = np.mean([brightness[r, f] for r in range(8) for f in range(8) if (r + f) % 2 == 0])
        dark_mean  = np.mean([brightness[r, f] for r in range(8) for f in range(8) if (r + f) % 2 == 1])

        print(f"Light mean: {light_mean:.1f}, Dark mean: {dark_mean:.1f}")

        for rank in range(8):
            for file in range(8):
                sq_idx = (7 - rank) * 8 + file
                b    = brightness[rank, file]
                std  = std_map[rank, file]
                edge = edge_map[rank, file]
                is_light_square = (rank + file) % 2 == 0

                if is_light_square:
                    # Use edge + texture; brightness is almost useless for white on white
                    edge_trigger     = edge > 0.115       # still separates from empty (~0.09–0.10)
                    texture_trigger  = std  > 20          # both b3/c2 have 31–34, empties ~10–15
                    occupancy[sq_idx] = edge_trigger and texture_trigger
                else:
                    # Keep conservative rule for dark squares
                    edge_trigger     = edge > 0.115
                    brightness_trigger = abs(b - dark_mean) > 24
                    occupancy[sq_idx] = edge_trigger and brightness_trigger



        self._draw_occupancy_overlay(inner, occupancy, std_map, brightness, edge_map, square_size)
        cv2.imwrite('../img/processed/warped_board.png', inner)
        return occupancy


    def _draw_occupancy_overlay(self, inner, occupancy, std_map, brightness, edge_map, square_size):
        overlay = cv2.cvtColor(inner, cv2.COLOR_GRAY2BGR)

        for rank in range(8):
            for file in range(8):
                sq_idx = (7 - rank) * 8 + file
                x = file * square_size
                y = rank * square_size
                cx = x + square_size // 2
                cy = y + square_size // 2

                if occupancy.get(sq_idx, False):
                    cv2.circle(overlay, (cx, cy), square_size // 3, (0, 255, 0), 2)
                else:
                    cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

                sq_name = chess.square_name(sq_idx)
                cv2.putText(overlay, sq_name, (x + 2, y + 11), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 0), 1)

                # ✅ Now shows edge density too for tuning
                cv2.putText(overlay, f"b:{brightness[rank,file]:.0f}", (x+2, y+square_size-20), cv2.FONT_HERSHEY_PLAIN, 0.55, (200,200,200), 1)
                cv2.putText(overlay, f"s:{std_map[rank,file]:.0f}",    (x+2, y+square_size-11), cv2.FONT_HERSHEY_PLAIN, 0.55, (200,200,200), 1)
                cv2.putText(overlay, f"e:{edge_map[rank,file]:.2f}",   (x+2, y+square_size-2),  cv2.FONT_HERSHEY_PLAIN, 0.55, (200,200,200), 1)

        cv2.imwrite('../img/processed/occupancy_overlay.png', overlay)
        print("Saved: occupancy_overlay.png")




    def occupancy_to_fen(self, occupancy):
        # Simple occupancy FEN (P/p not distinguished, use 'P' for occupied white side etc. - for showcase)
        board_str = ''
        empty_count = 0
        for sq in range(63, -1, -1):  # a8 to h1
            if sq not in occupancy or not occupancy[sq]:
                empty_count += 1
            else:
                if empty_count > 0:
                    board_str += str(empty_count)
                    empty_count = 0
                board_str += 'P'  # Placeholder for occupied
        if empty_count > 0:
            board_str += str(empty_count)
        fen = f"{board_str}/8 8 8 8 8 8 8 w - - 0 1"  # Dummy rest
        return fen

def showcase(image_path):
    recognizer = ChessBoardRecognizer()
    board_img, fen, occupancy = recognizer.recognize_from_image(image_path)
    
    if board_img is None:
        print("Failed to detect board")
        return
    
    # Show recognized board with corners
    cv2.imshow('Recognized Board', board_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print state
    print("Detected FEN:", fen)
    print("Occupied squares:", [chess.square_name(sq) for sq, occ in occupancy.items() if occ])
    
    # Save results
    cv2.imwrite('../img/processed/recognized_board.png', board_img)
    print("Saved: recognized_board.png, warped_board.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python showcase.py <image_path>")
    else:
        showcase(sys.argv[1])
