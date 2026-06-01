#include <Arduino_RouterBridge.h>
BridgeClass Bridge(Serial1);

#define X_STEP 2
#define X_DIR 5
#define Y_STEP 3
#define Y_DIR 6
#define Z_STEP 4
#define Z_DIR 7
#define ENABLE 8
#define BUTTON_OK A0
#define Z_ENDSTOP 11
#define GRIPPER_PIN 12

#define STEPS_PER_REV 50
#define MICROSTEP 1
#define STEP_DELAY_US 6000
#define Y_STEP_DELAY_MS 3000
#define Z_STEP_DELAY_MS 8000
#define Z_EXTRA_STEPS 100
#define X_ROTATE_STEPS 168
#define GRIPPER_CLOSED 2800
#define GRIPPER_OPEN 1000
#define MAX_MOVES 30

const unsigned long DEBOUNCE_DELAY = 50;
const unsigned long LONG_PRESS_TIME = 3000;
const unsigned long STATUS_MESSAGE_MS = 3000;

bool robotTurn = false;
bool gameActive = false;
bool setupComplete = false;
String robotNextMove = "----";
String statusMessage = "";
unsigned long statusMessageUntil = 0;

bool buttonOkState = false;
bool buttonOkLastState = false;
bool buttonOkPressedLatch = false;
bool buttonOkReleaseRequired = false;
unsigned long buttonOkDebounceTime = 0;
unsigned long buttonOkPressStart = 0;

bool calibrationDone = false;
bool verifyDone = false;
bool initialCaptureDone = false;

unsigned int gripperTargetPulse = 2200;

// Square sequence part
#define MAX_SEQ_STEPS 10

struct SquareStep {
  int  dirPin1, stepPin1;
  bool dir1;
  int  dirPin2, stepPin2; // 0,0 = single motor
  bool dir2;
  int  steps;
};

#define END { 0,0,false,0,0,false,0 }
////////////////////

void setStatusMessage(const String& message) {
  statusMessage = message;
  statusMessageUntil = millis() + STATUS_MESSAGE_MS;
  Serial.println(message);
}

String shortenMessage(String message) {
  message.replace("success:", "");
  message.replace("fail:", "");
  message.replace("unstable:", "");
  if (message.length() > 60) return message.substring(0, 60);
  return message;
}

bool callBridgeStep(const char* command, String& data) {
  data = "";
  Serial.print("[DEBUG] Bridge call -> ");
  Serial.println(command);
  bool ok = Bridge.call(command).result(data);
  Serial.print("[DEBUG] Bridge response <- ");
  Serial.println(data);
  if (!ok) {
    data = "fail:Bridge call failed";
    Serial.println("[DEBUG] Bridge call transport failed");
    return false;
  }
  return data.startsWith("success:");
}

void readButtons() {
  unsigned long currentTime = millis();
  bool okReading = (digitalRead(BUTTON_OK) == LOW);
  if (okReading != buttonOkLastState) {
    buttonOkDebounceTime = currentTime;
    if (okReading) buttonOkPressStart = currentTime;
  }
  if ((currentTime - buttonOkDebounceTime) > DEBOUNCE_DELAY && okReading != buttonOkState) {
    buttonOkState = okReading;
  }
  buttonOkLastState = okReading;
}

bool buttonOkPressed() {
  if (buttonOkReleaseRequired) {
    if (!buttonOkState) {
      buttonOkReleaseRequired = false;
      buttonOkPressedLatch = false;
    }
    return false;
  }

  if (buttonOkState && !buttonOkPressedLatch) {
    buttonOkPressedLatch = true;
    return true;
  }
  if (!buttonOkState) buttonOkPressedLatch = false;
  return false;
}

bool buttonOkLongPress() {
  if (!buttonOkReleaseRequired && buttonOkState && (millis() - buttonOkPressStart >= LONG_PRESS_TIME)) {
    buttonOkPressStart = millis() + 10000;
    return true;
  }
  return false;
}

void requireButtonRelease() {
  buttonOkReleaseRequired = true;
  buttonOkPressedLatch = true;
  buttonOkPressStart = millis();
}

void toggleTurn() {
  if (robotTurn) {
    robotTurn = false;
  } else {
    robotTurn = true;
  }
}

void logUiState() {
  if (statusMessage.length() > 0 && millis() < statusMessageUntil) {
    Serial.print("[STATUS] ");
    Serial.println(statusMessage);
  }
  if (gameActive) {
    Serial.print("[TURN] ");
    Serial.println(robotTurn ? "Robot" : "Player");
    Serial.print("[NEXT MOVE] ");
    Serial.println(robotNextMove);
  }
}

bool goToPiecePlace() {
  return true;
}

bool parseSquare(const char* square, char& fileOut, int& rankOut) {
  if (square == nullptr || strlen(square) < 2) return false;
  char file = square[0];
  if (file >= 'a' && file <= 'h') file -= 32;
  int rank = square[1] - '0';
  if (file < 'A' || file > 'H' || rank < 1 || rank > 8) return false;
  fileOut = file;
  rankOut = rank;
  return true;
}

void retractArmToHome() {
  initializePosition();
  rotateX90(false);
  setGripper(GRIPPER_CLOSED);
}

void performSquareAction(char file, int rank, unsigned int targetPulse) {
  gripperTargetPulse = targetPulse;
  goToSquare(file, rank);
}

bool pickupFromSquare(const char* square) {
  char file;
  int rank;
  if (!parseSquare(square, file, rank)) return false;
  setGripper(GRIPPER_OPEN);
  delay(500);
  performSquareAction(file, rank, GRIPPER_CLOSED);
  return true;
}

bool placeToSquare(const char* square) {
  char file;
  int rank;
  if (!parseSquare(square, file, rank)) return false;
  performSquareAction(file, rank, GRIPPER_OPEN);
  return true;
}

bool removeCapturedPiece(const char* square) {
  if (!pickupFromSquare(square)) return false;
  if (!goToPiecePlace()) return false;
  setGripper(GRIPPER_OPEN);
  delay(500);
  return true;
}

void executeUCIMove(const char* sq1, const char* sq2, int p1, int p2) {
  if (p2 != 0) {
    if (!removeCapturedPiece(sq2)) {
      setStatusMessage("Failed to remove captured piece");
      return;
    }
  }

  if (!pickupFromSquare(sq1)) {
    setStatusMessage("Failed to pick up source piece");
    return;
  }

  if (!placeToSquare(sq2)) {
    setStatusMessage("Failed to place piece on destination");
    return;
  }

  setGripper(GRIPPER_CLOSED);
  delay(500);
}

void startGame() {
  Bridge.call("reset_game");
  gameActive = true;
  setupComplete = true;
  robotTurn = true;
  robotNextMove = "----";
  statusMessage = "";
  requireButtonRelease();
  Bridge.call("log_event", "Game started. Robot is White.");
  Serial.println("[GAME] Started");
}

void endGame() {
  gameActive = false;
  Bridge.call("log_event", "Game resigned.");
  Serial.println("[GAME] Ended");
}

void requestAndExecuteRobotMove() {
  String data;
  Serial.println("[DEBUG] Requesting robot move");
  bool ok = Bridge.call("get_move").result(data);
  Serial.print("[DEBUG] get_move returned: ");
  Serial.println(data);
  if (!ok || data.length() == 0) return;

  char s1[5], s2[5];
  int p1, p2;
  if (sscanf(data.c_str(), "%[^,],%[^,],%d,%d", s1, s2, &p1, &p2) == 4) {
    robotNextMove = String(s1) + String(s2);
    Serial.print("[DEBUG] Executing robot move: ");
    Serial.println(robotNextMove);
    executeUCIMove(s1, s2, p1, p2);
    String refreshData;
    bool refreshSuccess = callBridgeStep("camera_refresh_reference", refreshData);
    if (!refreshSuccess) {
      setStatusMessage(shortenMessage(refreshData));
      Serial.print("[DEBUG] Reference refresh failed: ");
      Serial.println(refreshData);
      Bridge.call("log_event", "Robot move complete, but reference refresh failed.");
      return;
    }
    Serial.println("[DEBUG] Reference refresh succeeded");
    toggleTurn();
    requireButtonRelease();
    Bridge.call("log_event", "Robot move complete. Player turn.");
  }
}

void runSetupStep() {
  String data;
  if (!calibrationDone) {
    bool success = callBridgeStep("camera_calibrate", data);
    setStatusMessage(shortenMessage(data));
    calibrationDone = success;
    requireButtonRelease();
    return;
  }
  if (!verifyDone) {
    bool success = callBridgeStep("camera_verify", data);
    setStatusMessage(shortenMessage(data));
    verifyDone = success;
    requireButtonRelease();
    return;
  }
  if (!initialCaptureDone) {
    bool success = callBridgeStep("camera_capture_initial", data);
    setStatusMessage(shortenMessage(data));
    initialCaptureDone = success;
    if (success) {
      startGame();
    } else {
      requireButtonRelease();
    }
    return;
  }
}

void handlePlayerTurn() {
  if (buttonOkPressed() && gameActive && !robotTurn) {
    Serial.println("[DEBUG] Player requested move capture");
    String data;
    bool success = callBridgeStep("camera_capture_player_move", data);
    if (success) {
      setStatusMessage(String("Move ") + shortenMessage(data));
      Serial.print("[DEBUG] Player move accepted: ");
      Serial.println(data);
      toggleTurn();
      requireButtonRelease();
      Bridge.call("log_event", "Player move captured. Robot turn.");
    } else {
      Serial.print("[DEBUG] Player move capture failed: ");
      Serial.println(data);
      setStatusMessage(shortenMessage(data));
      requireButtonRelease();
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE, OUTPUT);
  pinMode(BUTTON_OK, INPUT_PULLUP);
  digitalWrite(ENABLE, LOW);
  Bridge.begin();

  pinMode(X_DIR, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(Z_ENDSTOP, INPUT_PULLUP);
  pinMode(GRIPPER_PIN, OUTPUT);

  delay(2000);
  initializePosition();
  rotateX90(false);
  setGripper(GRIPPER_CLOSED);
  setStatusMessage("Press OK to run setup");
  Serial.println("Merge system initialized");
}

void loop() {
  readButtons();

  if (!setupComplete) {
    if (buttonOkPressed()) {
      runSetupStep();
    }
    logUiState();
    delay(50);
    return;
  }

  if (gameActive && !robotTurn) {
    handlePlayerTurn();
  }

  if (gameActive && robotTurn) {
    requestAndExecuteRobotMove();
  }

  if (buttonOkLongPress() && gameActive) {
    endGame();
  }

  logUiState();
  delay(50);
}

struct Move {
  int  dirPin1, stepPin1;
  bool dir1;
  int  dirPin2, stepPin2;
  bool dir2;
  int  steps;
  bool isTwoMotors;

  Move() {}
  Move(int dp1, int sp1, bool d1, int dp2, int sp2, bool d2, int s, bool two)
    : dirPin1(dp1), stepPin1(sp1), dir1(d1),
      dirPin2(dp2), stepPin2(sp2), dir2(d2),
      steps(s), isTwoMotors(two) {}
};
Move moveQueue[MAX_MOVES];
int  moveCount = 0;

void rememberMove(int dirPin1, int stepPin1, int steps, bool clockwise);
void rememberMove(int dirPin1, int stepPin1, bool dir1,
                  int dirPin2, int stepPin2, bool dir2,
                  int steps);
void goHome();


// ─── RECORDABLE MOTIONS ──────────────────────────────────────────────────────

void rememberMove(int dirPin1, int stepPin1, int steps, bool clockwise) {
  if (moveCount >= MAX_MOVES) return;
  moveQueue[moveCount++] = Move(dirPin1, stepPin1, clockwise, 0, 0, false, steps, false);
}

void rememberMove(int dirPin1, int stepPin1, bool dir1,
                  int dirPin2, int stepPin2, bool dir2,
                  int steps) {
  if (moveCount >= MAX_MOVES) return;
  moveQueue[moveCount++] = Move(dirPin1, stepPin1, dir1, dirPin2, stepPin2, dir2, steps, true);
}

void goHome() {
  for (int i = moveCount - 1; i >= 0; i--) {
    Move& m = moveQueue[i];
    if (m.isTwoMotors) {
      stepTwoMotors(m.dirPin1, m.stepPin1, !m.dir1,
                    m.dirPin2, m.stepPin2, !m.dir2,
                    m.steps);
    } else {
      stepMotor(m.dirPin1, m.stepPin1, m.steps, !m.dir1);
    }
  }
  moveCount = 0;
  rotateX90(false);
  delay(300);
}


// ─── MOTORS ──────────────────────────────────────────────────────────────────

void setGripper(unsigned int pulse) {
  for (unsigned char i = 0; i < 8; i++) {
    digitalWrite(GRIPPER_PIN, HIGH);
    delayMicroseconds(pulse);
    digitalWrite(GRIPPER_PIN, LOW);
  }
}

void stepMotor(int dirPin, int stepPin, int steps, bool clockwise) {
  digitalWrite(dirPin, clockwise ? HIGH : LOW);
  delay(1);
  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}

void stepTwoMotors(int dirPin1, int stepPin1, bool dir1,
                   int dirPin2, int stepPin2, bool dir2,
                   int steps) {
  digitalWrite(dirPin1, dir1 ? HIGH : LOW);
  digitalWrite(dirPin2, dir2 ? HIGH : LOW);
  delay(1);
  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(stepPin1, HIGH);
    digitalWrite(stepPin2, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(stepPin1, LOW);
    digitalWrite(stepPin2, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}


// ─── POSITION ────────────────────────────────────────────────────────────────

void initializePosition() {
  digitalWrite(Y_DIR, LOW);
  digitalWrite(Z_DIR, HIGH);

  int zCounter = 0;

  while (digitalRead(Z_ENDSTOP) == LOW) {
    digitalWrite(Y_STEP, HIGH);
    delayMicroseconds(Y_STEP_DELAY_MS);
    digitalWrite(Y_STEP, LOW);
    delayMicroseconds(Y_STEP_DELAY_MS);

    zCounter++;
    if (zCounter >= 3) {
      digitalWrite(Z_STEP, HIGH);
      delayMicroseconds(Z_STEP_DELAY_MS);
      digitalWrite(Z_STEP, LOW);
      delayMicroseconds(Z_STEP_DELAY_MS);
      zCounter = 0;
    }
  }

  digitalWrite(Y_DIR, HIGH);
  for (int i = 0; i < 20; i++) {
    digitalWrite(Y_STEP, HIGH);
    delayMicroseconds(Y_STEP_DELAY_MS);
    digitalWrite(Y_STEP, LOW);
    delayMicroseconds(Y_STEP_DELAY_MS);
  }

  for (int i = 0; i < Z_EXTRA_STEPS; i++) {
    digitalWrite(Z_STEP, HIGH);
    delayMicroseconds(Z_STEP_DELAY_MS);
    digitalWrite(Z_STEP, LOW);
    delayMicroseconds(Z_STEP_DELAY_MS);
  }
}

void rotateX90(bool clockwise) {
  digitalWrite(X_DIR, clockwise ? HIGH : LOW);
  delay(1);
  for (int i = 0; i < X_ROTATE_STEPS; i++) {
    digitalWrite(X_STEP, HIGH);
    delayMicroseconds(10000);
    digitalWrite(X_STEP, LOW);
    delayMicroseconds(10000);
  }
}


// ─── SQUARE SEQUENCES ────────────────────────────────────────────────────────

static const SquareStep squareSeqs[8][8][MAX_SEQ_STEPS] = {
  // A1-A8
  {
    {{ X_DIR,X_STEP,true,  0,0,false,120 }, { Z_DIR,Z_STEP,true,  0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 90 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false,104 }, { Z_DIR,Z_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,107 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 10 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 91 }, { Z_DIR,Z_STEP,true,  0,0,false, 55 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,107 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 20 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 79 }, { Z_DIR,Z_STEP,true,  0,0,false, 75 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,115 }, { Z_DIR,Z_STEP,false,0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 71 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 40 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 40 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 65 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 95 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 58 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 33 }, { Z_DIR,Z_STEP,true,0,0,false,70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,60 }, END },
  },
  // B1-B8
  {
    {{ X_DIR,X_STEP,true,  0,0,false, 94 }, { Z_DIR,Z_STEP,true,  0,0,false, 30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 40 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 50 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 80 }, { Z_DIR,Z_STEP,true,  0,0,false, 30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 66 }, { Z_DIR,Z_STEP,true,  0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 90 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 57 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,105 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 58 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 30 }, { Z_DIR,Z_STEP,false,0,0,false,  8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 27 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 53 }, { Z_DIR,Z_STEP,true,  0,0,false, 90 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false, 40 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,false,0,0,false,8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,10 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 44 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 80 }, { Z_DIR,Z_STEP,false,0,0,false,10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,15 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 42 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 90 }, { Z_DIR,Z_STEP,false,0,0,false,10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,10 }, END },
  },
  // C1-C8
  {
    {{ X_DIR,X_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 50 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 55 }, { Z_DIR,Z_STEP,true,  0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 85 }, { Z_DIR,Z_STEP,false,0,0,false, 30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 40 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 43 }, { Z_DIR,Z_STEP,true,  0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,100 }, { Z_DIR,Z_STEP,false,0,0,false, 22 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 20 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 38 }, { Z_DIR,Z_STEP,true,  0,0,false, 65 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,115 }, { Z_DIR,Z_STEP,false,0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 12 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 20 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 32 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 50 }, { Z_DIR,Z_STEP,false,0,0,false,8 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,50 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 30 }, { Z_DIR,Z_STEP,true,  0,0,false, 90 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false,10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 30 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 80 }, { Z_DIR,Z_STEP,true,0,0,false,30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,10 }, END },
  },
  // D1-D8
  {
    {{ X_DIR,X_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 60 }, { Z_DIR,Z_STEP,false,0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 22 }, { Z_DIR,Z_STEP,true,  0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 18 }, { Z_DIR,Z_STEP,true,  0,0,false, 40 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 90 }, { Z_DIR,Z_STEP,false,0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 15 }, { Z_DIR,Z_STEP,true,  0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,110 }, { Z_DIR,Z_STEP,false,0,0,false, 18 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 30 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 16 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,120 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 28 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 14 }, { Z_DIR,Z_STEP,true,  0,0,false, 85 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 15 }, { Z_DIR,Z_STEP,true,  0,0,false, 95 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,true,  0,0,false, 14 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 60 }, END },
  },
  // E1-E8
  {
    {{ X_DIR,X_STEP,false, 0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 65 }, { Z_DIR,Z_STEP,false,0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 15 }, { Z_DIR,Z_STEP,true,  0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 10 }, { Z_DIR,Z_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 85 }, { Z_DIR,Z_STEP,false,0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false,  5 }, { Z_DIR,Z_STEP,true,  0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,105 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ Z_DIR,Z_STEP,true,  0,0,false, 65 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,115 }, { Z_DIR,Z_STEP,true, 0,0,false, 15 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 33 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false,  3 }, { Z_DIR,Z_STEP,true,  0,0,false, 80 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,120 }, { Z_DIR,Z_STEP,true, 0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false,  3 }, { Z_DIR,Z_STEP,true,  0,0,false, 90 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false,  3 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 15 }, { Z_DIR,Z_STEP,true,0,0,false,15 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,52 }, END },
  },
  // F1-F8
  {
    {{ X_DIR,X_STEP,false, 0,0,false, 55 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 65 }, { Z_DIR,Z_STEP,false,0,0,false, 30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 45 }, { Z_DIR,Z_STEP,true,  0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 33 }, { Z_DIR,Z_STEP,true,  0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 85 }, { Z_DIR,Z_STEP,false,0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 25 }, { Z_DIR,Z_STEP,true,  0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,100 }, { Z_DIR,Z_STEP,false,0,0,false, 15 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 22 }, { Z_DIR,Z_STEP,true,  0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,115 }, { Z_DIR,Z_STEP,true, 0,0,false, 12 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 15 }, { Z_DIR,Z_STEP,true,  0,0,false, 85 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,120 }, { Z_DIR,Z_STEP,true, 0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 12 }, { Z_DIR,Z_STEP,true,  0,0,false, 95 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 50 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 14 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 40 }, { Z_DIR,Z_STEP,true,0,0,false,20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,30 }, END },
  },
  // G1-G8
  {
    {{ X_DIR,X_STEP,false, 0,0,false, 88 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 65 }, { Z_DIR,Z_STEP,false,0,0,false, 15 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 63 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 70 }, { Z_DIR,Z_STEP,true,  0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,false,0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 55 }, { Z_DIR,Z_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true, 0,0,false,  3 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 50 }, { Z_DIR,Z_STEP,true,  0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 95 }, { Z_DIR,Z_STEP,true, 0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 40 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 40 }, { Z_DIR,Z_STEP,true,  0,0,false, 65 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,110 }, { Z_DIR,Z_STEP,true, 0,0,false, 20 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 36 }, { Z_DIR,Z_STEP,true,  0,0,false, 80 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,120 }, { Z_DIR,Z_STEP,true, 0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 35 }, { Z_DIR,Z_STEP,true,  0,0,false, 90 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 80 }, { Z_DIR,Z_STEP,true, 0,0,false, 30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true,0,0,false,30 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,10 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 28 }, { Z_DIR,Z_STEP,true,  0,0,false, 90 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 80 }, { Z_DIR,Z_STEP,true, 0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 75 }, { Z_DIR,Z_STEP,true,0,0,false,50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,30 }, END },
  },
  // H1-H8
  {
    {{ X_DIR,X_STEP,false, 0,0,false,110 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 65 }, { Z_DIR,Z_STEP,true, 0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 63 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 90 }, { Z_DIR,Z_STEP,true,  0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 70 }, { Z_DIR,Z_STEP,true, 0,0,false,  5 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 55 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 75 }, { Z_DIR,Z_STEP,true,  0,0,false, 35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 85 }, { Z_DIR,Z_STEP,true, 0,0,false, 10 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 60 }, { Z_DIR,Z_STEP,true,  0,0,false, 50 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 95 }, { Z_DIR,Z_STEP,true, 0,0,false, 25 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 45 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 55 }, { Z_DIR,Z_STEP,true,  0,0,false, 65 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,110 }, { Z_DIR,Z_STEP,true, 0,0,false, 40 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 45 }, { Z_DIR,Z_STEP,true,  0,0,false, 80 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,120 }, { Z_DIR,Z_STEP,true, 0,0,false, 45 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 35 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 43 }, { Z_DIR,Z_STEP,true,  0,0,false, 95 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 60 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 50 }, END },
    {{ X_DIR,X_STEP,false, 0,0,false, 42 }, { Z_DIR,Z_STEP,true,  0,0,false,100 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,125 }, { Z_DIR,Z_STEP,true, 0,0,false, 70 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true, 40 }, { Z_DIR,Z_STEP,true,0,0,false,35 }, { Z_DIR,Z_STEP,false,Y_DIR,Y_STEP,true,45 }, END },
  },
};

void goToSquare(char col, int row) {
  moveCount = 0;
  rotateX90(true);

  // Make sure to pass the cases to upper case index
  if (col >= 'a' && col <= 'h') {
    col = col - 32;
  }
  // Turn the index number into 0-7
  int ci = col - 'A';
  int ri = row - 1;
  // Re check if the index is valid
  if (ci < 0 || ci > 7 || ri < 0 || ri > 7) return;

  // Repeated code from the original goto functions
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  // Pointer to get the list of steps of the square
  const SquareStep* seq = squareSeqs[ci][ri];

  // Go through each step and execute it
  for (int i = 0; i < MAX_SEQ_STEPS && seq[i].steps != 0; i++) {
    const SquareStep& s = seq[i];
    if (s.dirPin2 == 0) {
      rememberMove(s.dirPin1, s.stepPin1, s.steps, s.dir1);
      stepMotor(s.dirPin1, s.stepPin1, s.steps, s.dir1);
    } else {
      rememberMove(s.dirPin1, s.stepPin1, s.dir1, s.dirPin2, s.stepPin2, s.dir2, s.steps);
      stepTwoMotors(s.dirPin1, s.stepPin1, s.dir1, s.dirPin2, s.stepPin2, s.dir2, s.steps);
    }
    delay(500);
  }

  setGripper(gripperTargetPulse);
  goHome();
}
