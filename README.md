# Hello World! Chess Bot

A chess-playing robot arm built with Python and C++ (Arduino). It uses computer vision to read the board, Stockfish to calculate the best move, and a 3-axis robot arm to physically execute it — completely autonomously after setup.

## How It Works

The system runs two communication layers:

- **Software layer** — Python handles board state, camera vision (OpenCV), and move generation via Stockfish. Commands like `get_move`, `camera_calibrate`, and `camera_capture_player_move` are exposed to the Arduino via a bridge.
- **Hardware layer** — The Arduino translates logical moves into stepper/servo motor signals, controlling the arm across 3 axes to pick up and place pieces.

**Game loop:**
1. Camera calibrates and captures the initial board state
2. Robot plays as White; player makes a move and presses OK
3. Camera detects the new board state and sends it to Stockfish
4. The arm executes the best move
5. Repeat until checkmate or resignation

## Requirements

### Hardware
| Component | Notes |
|---|---|
| [Arduino UNO Q (2GB/4GB)](https://www.arduino.cc/product-uno-q) | Main controller |
| Arduino CNC Shield + 4× DRV8825 drivers | Motor control |
| 4× NEMA 17 Stepper Motors | Arm movement |
| 1× SG90 Micro Servo | Gripper |
| 3× Endstop Microswitches | Homing |
| Webcam (1080p wide-angle) | Board detection |
| 12V 10A Power Supply | Powers motors |
| 3D-printed arm parts | See [/Arm print](https://github.com/Nannerscube/HelloWorld/tree/main/Arm%20print) |

Total hardware cost: ~€203 + 3D printing

### Software & Libraries

**Arduino:**
- `Arduino_RouterBridge`

**Python (`requirements.txt`):**
```
chess
numpy
opencv-python
```

**External:**
- [Stockfish](https://stockfishchess.org/download/) — place the executable in the AppLabs Python folder, or set the `STOCKFISH_PATH` environment variable

### Optional
- [Visual Studio Code](https://code.visualstudio.com/download)

## Installation

1. Download the ZIP from GitHub and extract the contents onto the Arduino UNO Q
2. Install Python dependencies: `pip install -r requirements.txt`
3. Download Stockfish and place the executable in the AppLabs Python folder

## Configuration

- `ENABLE_PIN = 8`
- Endstops and motor step/dir pins are defined in `sketch.ino`

## Startup Guide

1. Assemble the robot arm — mount motors, gripper, camera, and chessboard in fixed positions
2. Wire Arduino, CNC shield, motors, endstops, button, and servo per the pin mapping; ensure common ground across all components
3. Open AppLabs and create a new project
4. Install the `Arduino_RouterBridge` library and upload `sketch.ino` to the board
5. In the Python environment, ensure `main.py` and `requirements.txt` exist with the required libraries
6. Click **Start** and confirm the bridge reports chess and vision systems are working in the console
7. Power the robot, clear the work area, and press **OK** once — the arm will home itself
8. Press **OK** again to calibrate the camera and verify the board warp
9. Place chess pieces on the board with the robot's side as White
10. Press **OK** again to capture the initial board state — the game begins
11. **Your turn:** make a legal move and press **OK** so the camera captures it
12. **Robot's turn:** wait for the arm to finish and the reference image to refresh before touching the board
13. To stop the game, hold **OK** for ~10 seconds to trigger the resignation flow

## Known Limitations

| Case | Description |
|---|---|
| Uneven lighting | Camera may fail to detect board state correctly |
| Shadows on the board | Can cause false move detections |
| Lighting changes mid-game | Breaks reference frame comparison, causing incorrect detections |
| Castling desync | Engine passes `from_square, to_square` only — rook movement can desync the virtual and physical boards |
| No human promotion support | If the human player reaches promotion, the system has no handler for it |
| No promotion display | Engine promotions are not visually indicated |
| Physical board movement | If the board shifts during play, detection accuracy degrades |

## Project Structure

```
HelloWorld/
├── Arm print/         # 3D print files for the robot arm
├── Python/
│   ├── main.py        # Entry point
│   └── requirements.txt
├── sketch.ino         # Arduino firmware
└── tests/
    ├── grippertest.ino
    └── ...
```

## Acknowledgements

- [Stockfish](https://stockfishchess.org/) — chess engine
- [python-chess](https://python-chess.readthedocs.io/) — board logic
- [OpenCV](https://opencv.org/) — computer vision
