#include <AccelStepper.h>
#include <Servo.h>
#include <Arduino_RouterBridge.h>
#include <LCDi2c.h>

/*THIS IS group LEON's CODE BUT WILL BE REWORKED ONLY FOR DOCUMETATION*/

#define X_STEP 2
#define X_DIR 5
#define Y_STEP 3
#define Y_DIR 6
#define Z_STEP 4
#define Z_DIR 7
#define ENABLE 8
#define BUTTON_UP 11
#define BUTTON_OK 10
#define BUTTON_DOWN 9
#define GRIPPER_PIN 12

#define GRIPPER_OPEN 25
#define GRIPPER_CLOSED 60
#define Z_TRAVEL_POS 0
#define X_MAX_SPEED 25000.0
#define X_ACCEL 10000.0
#define Y_MAX_SPEED 30000.0
#define Y_ACCEL 12000.0
#define Z_MAX_SPEED 35000.0
#define Z_ACCEL 15000.0

AccelStepper stepperX(AccelStepper::DRIVER, X_STEP, X_DIR);
AccelStepper stepperY(AccelStepper::DRIVER, Y_STEP, Y_DIR);
AccelStepper stepperZ(AccelStepper::DRIVER, Z_STEP, Z_DIR);
Servo gripperServo;
LCDi2c lcd(0x27, Wire);

enum MenuState {
    MENU_SETUP_CALIBRATE,
    MENU_SETUP_VERIFY,
    MENU_SETUP_CAPTURE,
    MENU_START,
    MENU_CONFIRM_START,
    MENU_GAME,
    MENU_CONFIRM_RESIGN
};

MenuState currentMenu = MENU_SETUP_CALIBRATE;
bool robotTurn = false;
bool gameActive = false;
int menuSelection = 0;
String robotNextMove = "----";
String statusMessage = "";
unsigned long statusMessageUntil = 0;

#define STARTING_TIME 1800000
unsigned long playerStartTime, robotStartTime, playerElapsedTime, robotElapsedTime;
unsigned long playerPauseTime, robotPauseTime, lastDisplayUpdate;
const unsigned long DISPLAY_UPDATE_INTERVAL = 100;
const unsigned long STATUS_MESSAGE_MS = 3000;

int playerScore = 0;
int robotScore = 0;

bool buttonUpState, buttonOkState, buttonDownState;
bool buttonUpLastState, buttonOkLastState, buttonDownLastState;
unsigned long buttonUpDebounceTime, buttonOkDebounceTime, buttonDownDebounceTime, buttonOkPressStart, buttonDownPressStart;
const unsigned long DEBOUNCE_DELAY = 50, LONG_PRESS_TIME = 3000;

struct Pos { long x; long y; };
const Pos squareTable[8][8] = {
    /* A */ { {-4600L,-21250L}, {-4220L,-20500L}, {-4010L,-19770L}, {-3960L,-18980L}, {-4030L,-18140L}, {-4210L,-17200L}, {-4550L,-16030L}, {-5440L,-13700L} },
    /* B */ { {-4040L,-21540L}, {-3610L,-20660L}, {-3430L,-19840L}, {-3430L,-19040L}, {-3530L,-18230L}, {-3760L,-17320L}, {-4070L,-16230L}, {-4660L,-14670L} },
    /* C */ { {-3170L,-21530L}, {-2800L,-20590L}, {-2730L,-19750L}, {-2830L,-18960L}, {-3030L,-18180L}, {-3290L,-17320L}, {-3650L,-16300L}, {-4200L,-14940L} },
    /* D */ { {-1500L,-20800L}, {-1750L,-20040L}, {-1990L,-19350L}, {-2250L,-18650L}, {-2540L,-17930L}, {-2880L,-17130L}, {-3720L,-16170L}, {-3820L,-14930L} },
    /* E */ { {-7530L, -4610L}, {-7420L, -5370L}, {-7210L, -6090L}, {-6940L, -6800L}, {-6640L, -7500L}, {-6310L, -8320L}, {-5910L, -9240L}, {-5410L,-10430L} },
    /* F */ { {-6110L, -4070L}, {-6430L, -4960L}, {-6480L, -5770L}, {-6360L, -6560L}, {-6160L, -7340L}, {-5890L, -8180L}, {-5540L, -9150L}, {-5030L,-10420L} },
    /* G */ { {-5170L, -4010L}, {-5590L, -4840L}, {-5760L, -5640L}, {-5760L, -6430L}, {-5650L, -7250L}, {-5430L, -8120L}, {-5100L, -9210L}, {-4550L,-10690L} },
    /* H */ { {-4570L, -4310L}, {-4940L, -5040L}, {-5140L, -5750L}, {-5200L, -6530L}, {-5120L, -7380L}, {-4950L, -8300L}, {-4630L, -9410L}, {-3660L,-11940L} },
};
const Pos HOME = {0L, 0L}, SAFE_POS = {-1000L, -6200L}, PIECE_PLACE = { -4010L, -7770L };

struct Piece { int id; long pickupHeight; long placeHeight; };
Piece pieces[] = { {0,0,0}, {1,38000,37500}, {2,38000,37500}, {3,36400,35900}, {4,36400,35900}, {5,34000,33500}, {6,34000,33500} };

void setStatusMessage(const String& message) {
    statusMessage = message;
    statusMessageUntil = millis() + STATUS_MESSAGE_MS;
    Serial.println(message);
}

String shortenMessage(String message) {
    message.replace("success:", "");
    message.replace("fail:", "");
    message.replace("unstable:", "");
    if (message.length() > 20) return message.substring(0, 20);
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

void setup() {
    Serial.begin(115200);
    pinMode(ENABLE, OUTPUT); digitalWrite(ENABLE, LOW);
    pinMode(BUTTON_UP, INPUT_PULLUP); pinMode(BUTTON_OK, INPUT_PULLUP); pinMode(BUTTON_DOWN, INPUT_PULLUP);

    gripperServo.attach(GRIPPER_PIN);
    gripperServo.write(GRIPPER_CLOSED);

    stepperX.setMaxSpeed(X_MAX_SPEED); stepperX.setAcceleration(X_ACCEL);
    stepperY.setMaxSpeed(Y_MAX_SPEED); stepperY.setAcceleration(Y_ACCEL);
    stepperZ.setMaxSpeed(Z_MAX_SPEED); stepperZ.setAcceleration(Z_ACCEL);

    lcd.begin(4, 20);
    lcd.cls();
    Bridge.begin();
    displaySetupCalibrate();
    Serial.println("Chess Robot System Initialized");
}

void loop() {
    readButtons();

    switch (currentMenu) {
        case MENU_SETUP_CALIBRATE: handleSetupCalibrate(); break;
        case MENU_SETUP_VERIFY: handleSetupVerify(); break;
        case MENU_SETUP_CAPTURE: handleSetupCapture(); break;
        case MENU_START: handleStartMenu(); break;
        case MENU_CONFIRM_START: handleConfirmStartMenu(); break;
        case MENU_GAME: handleGameScreen(); checkForResignHold(); break;
        case MENU_CONFIRM_RESIGN: handleConfirmResignMenu(); break;
    }

    if (gameActive && robotTurn) requestAndExecuteRobotMove();

    if (millis() - lastDisplayUpdate >= DISPLAY_UPDATE_INTERVAL) {
        updateDisplay();
        lastDisplayUpdate = millis();
    }
}

void handleSetupCalibrate() {
    if (buttonOkPressed()) {
        String data;
        bool success = callBridgeStep("camera_calibrate", data);
        setStatusMessage(shortenMessage(data));
        if (success) {
            currentMenu = MENU_SETUP_VERIFY;
            displaySetupVerify();
        } else {
            displaySetupCalibrate();
        }
    }
}

void handleSetupVerify() {
    if (buttonOkPressed()) {
        String data;
        bool success = callBridgeStep("camera_verify", data);
        setStatusMessage(shortenMessage(data));
        if (success) {
            currentMenu = MENU_SETUP_CAPTURE;
            displaySetupCapture();
        } else {
            displaySetupVerify();
        }
    }
}

void handleSetupCapture() {
    if (buttonOkPressed()) {
        String data;
        bool success = callBridgeStep("camera_capture_initial", data);
        setStatusMessage(shortenMessage(data));
        if (success) {
            currentMenu = MENU_START;
            displayStartMenu();
        } else {
            displaySetupCapture();
        }
    }
}

void handleStartMenu() {
    if (buttonOkPressed()) {
        currentMenu = MENU_CONFIRM_START;
        menuSelection = 0;
        displayConfirmStartMenu();
    }
}

void handleConfirmStartMenu() {
    if (buttonUpPressed() || buttonDownPressed()) {
        menuSelection = 1 - menuSelection;
        displayConfirmStartMenu();
    }
    if (buttonOkPressed()) {
        if (menuSelection == 0) {
            startGame();
        } else {
            currentMenu = MENU_START;
            menuSelection = 0;
            displayStartMenu();
        }
    }
}

void handleGameScreen() {
    if (buttonOkPressed() && gameActive && !robotTurn) {
        Serial.println("[DEBUG] Player requested move capture");
        String data;
        bool success = callBridgeStep("camera_capture_player_move", data);
        if (success) {
            setStatusMessage(String("Move ") + shortenMessage(data));
            Serial.print("[DEBUG] Player move accepted: ");
            Serial.println(data);
            toggleTurn();
            Bridge.call("log_event", "Player move captured. Robot turn.");
        } else {
            Serial.print("[DEBUG] Player move capture failed: ");
            Serial.println(data);
            setStatusMessage(shortenMessage(data));
        }
    }

    if (gameActive) {
        if (getPlayerTime() == 0) Serial.println("Player time out - Robot wins!");
        if (getRobotTime() == 0) Serial.println("Robot time out - Player wins!");
    }

    updateTimers();
}

void checkForResignHold() {
    if (buttonDownLongPress() && gameActive) {
        currentMenu = MENU_CONFIRM_RESIGN;
        menuSelection = 0;
        pauseTimers();
        displayConfirmResignMenu();
    }
}

void handleConfirmResignMenu() {
    if (buttonUpPressed() || buttonDownPressed()) {
        menuSelection = 1 - menuSelection;
        displayConfirmResignMenu();
    }
    if (buttonOkPressed()) {
        if (menuSelection == 0) {
            endGame();
        } else {
            currentMenu = MENU_GAME;
            menuSelection = 0;
            resumeTimers();
            displayGameScreen();
        }
    }
}

void startGame() {
    Bridge.call("reset_game");

    gameActive = true;
    robotTurn = true;
    currentMenu = MENU_GAME;
    playerScore = 0;
    robotScore = 0;
    playerElapsedTime = 0;
    robotElapsedTime = 0;
    robotStartTime = millis();
    playerStartTime = 0;
    robotNextMove = "----";
    statusMessage = "";
    displayGameScreen();
    Bridge.call("log_event", "Game started. Robot is White.");
}

void endGame() {
    gameActive = false;
    currentMenu = MENU_START;
    menuSelection = 0;
    displayStartMenu();
    Bridge.call("log_event", "Game resigned.");
}

void toggleTurn() {
    unsigned long now = millis();
    if (robotTurn) {
        robotTurn = false;
        if (robotStartTime > 0) robotElapsedTime += (now - robotStartTime);
        playerStartTime = now;
    } else {
        robotTurn = true;
        if (playerStartTime > 0) playerElapsedTime += (now - playerStartTime);
        robotStartTime = now;
    }
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
        displayGameScreen();
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
        Bridge.call("log_event", "Robot move complete. Player turn.");
    }
}

void executeUCIMove(const char* sq1, const char* sq2, int p1, int p2) {
    Pos t1, t2;
    getSquarePos(sq1, t1);
    getSquarePos(sq2, t2);

    if (p2 != 0) {
        moveXY(t2.x, t2.y); gripperServo.write(GRIPPER_OPEN); delay(800);
        moveZ(pieces[p2].pickupHeight); gripperServo.write(GRIPPER_CLOSED); delay(800);
        moveZ(0); moveXY(PIECE_PLACE.x, PIECE_PLACE.y); moveZ(pieces[p2].placeHeight);
        gripperServo.write(GRIPPER_OPEN); delay(800); moveZ(0);
    }

    moveXY(t1.x, t1.y); gripperServo.write(GRIPPER_OPEN); delay(800);
    moveZ(pieces[p1].pickupHeight); gripperServo.write(GRIPPER_CLOSED); delay(800);
    moveZ(0); moveXY(t2.x, t2.y); moveZ(pieces[p1].placeHeight);
    gripperServo.write(GRIPPER_OPEN); delay(800);
    moveZ(0);
    gripperServo.write(GRIPPER_CLOSED); delay(800);
    moveToHome();
}

void moveXY(long x, long y) {
    stepperX.moveTo(x);
    stepperY.moveTo(y);
    while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0) {
        stepperX.run();
        stepperY.run();
    }
}

void moveZ(long z) {
    stepperZ.moveTo(z);
    while (stepperZ.distanceToGo() != 0) stepperZ.run();
}

void moveToHome() {
    // First move to safe position
    moveXY(SAFE_POS.x, SAFE_POS.y);
    // Then move to home
    moveXY(HOME.x, HOME.y);
}

bool getSquarePos(const char* s, Pos& out) {
    char f = (s[0] >= 'a') ? s[0] - 32 : s[0];
    int r = s[1] - '1';
    int c = f - 'A';
    if (c < 0 || c > 7 || r < 0 || r > 7) return false;
    out = squareTable[c][r];
    return true;
}

void updateTimers() {
}

void pauseTimers() {
    unsigned long now = millis();
    if (robotTurn && robotStartTime > 0) {
        robotPauseTime = now - robotStartTime;
    } else if (!robotTurn && playerStartTime > 0) {
        playerPauseTime = now - playerStartTime;
    }
}

void resumeTimers() {
    unsigned long now = millis();
    if (robotTurn) {
        robotStartTime = now - robotPauseTime;
    } else {
        playerStartTime = now - playerPauseTime;
    }
}

unsigned long getPlayerTime() {
    unsigned long elapsed;
    if (gameActive && !robotTurn && playerStartTime > 0) elapsed = playerElapsedTime + (millis() - playerStartTime);
    else elapsed = playerElapsedTime;
    return (elapsed >= STARTING_TIME) ? 0 : STARTING_TIME - elapsed;
}

unsigned long getRobotTime() {
    unsigned long elapsed;
    if (gameActive && robotTurn && robotStartTime > 0) elapsed = robotElapsedTime + (millis() - robotStartTime);
    else elapsed = robotElapsedTime;
    return (elapsed >= STARTING_TIME) ? 0 : STARTING_TIME - elapsed;
}

void readButtons() {
    unsigned long currentTime = millis();

    bool upReading = (digitalRead(BUTTON_UP) == LOW);
    if (upReading != buttonUpLastState) buttonUpDebounceTime = currentTime;
    if ((currentTime - buttonUpDebounceTime) > DEBOUNCE_DELAY && upReading != buttonUpState) buttonUpState = upReading;
    buttonUpLastState = upReading;

    bool downReading = (digitalRead(BUTTON_DOWN) == LOW);
    if (downReading != buttonDownLastState) {
        buttonDownDebounceTime = currentTime;
        if (downReading) buttonDownPressStart = currentTime;
    }
    if ((currentTime - buttonDownDebounceTime) > DEBOUNCE_DELAY && downReading != buttonDownState) buttonDownState = downReading;
    buttonDownLastState = downReading;

    bool okReading = (digitalRead(BUTTON_OK) == LOW);
    if (okReading != buttonOkLastState) {
        buttonOkDebounceTime = currentTime;
        if (okReading) buttonOkPressStart = currentTime;
    }
    if ((currentTime - buttonOkDebounceTime) > DEBOUNCE_DELAY && okReading != buttonOkState) buttonOkState = okReading;
    buttonOkLastState = okReading;
}

bool buttonUpPressed() {
    static bool lastState = false;
    if (buttonUpState && !lastState) {
        lastState = true;
        return true;
    }
    if (!buttonUpState) lastState = false;
    return false;
}

bool buttonDownPressed() {
    static bool lastState = false;
    if (buttonDownState && !lastState) {
        lastState = true;
        return true;
    }
    if (!buttonDownState) lastState = false;
    return false;
}

bool buttonDownLongPress() {
    if (buttonDownState && (millis() - buttonDownPressStart >= LONG_PRESS_TIME)) {
        buttonDownPressStart = millis() + 10000;
        return true;
    }
    return false;
}

bool buttonOkPressed() {
    static bool lastState = false;
    if (buttonOkState && !lastState) {
        lastState = true;
        return true;
    }
    if (!buttonOkState) lastState = false;
    return false;
}

bool buttonOkLongPress() {
    if (buttonOkState && (millis() - buttonOkPressStart >= LONG_PRESS_TIME)) {
        buttonOkPressStart = millis() + 10000;
        return true;
    }
    return false;
}

void displaySetupCalibrate() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf(" Camera Setup");
    lcd.locate(2, 1); lcd.printf("Empty board");
    lcd.locate(3, 1); lcd.printf("OK = calibrate");
    lcd.locate(4, 1); lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
}

void displaySetupVerify() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf(" Verify Camera");
    lcd.locate(2, 1); lcd.printf("Check live view");
    lcd.locate(3, 1); lcd.printf("OK = verify");
    lcd.locate(4, 1); lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
}

void displaySetupCapture() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf(" Initial Board");
    lcd.locate(2, 1); lcd.printf("Place pieces");
    lcd.locate(3, 1); lcd.printf("OK = capture");
    lcd.locate(4, 1); lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
}

void displayStartMenu() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf(" Setup complete");
    lcd.locate(2, 1); lcd.printf(" > Start Game <");
    lcd.locate(3, 1); lcd.printf(" Press OK");
    lcd.locate(4, 1); lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
}

void displayConfirmStartMenu() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf(" Start the game?");
    lcd.locate(3, 1); lcd.printf(menuSelection == 0 ? "      > Yes <" : "        Yes");
    lcd.locate(4, 1); lcd.printf(menuSelection == 1 ? "      > No <" : "        No");
}

void displayConfirmResignMenu() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf("   Resign game?");
    lcd.locate(3, 1); lcd.printf(menuSelection == 0 ? "      > Yes <" : "        Yes");
    lcd.locate(4, 1); lcd.printf(menuSelection == 1 ? "      > No <" : "        No");
}

void displayGameScreen() {
    lcd.cls();
    lcd.locate(1, 1); lcd.printf("You  ");
    lcd.locate(1, 9); lcd.printf("%-4s", robotNextMove.c_str());
    lcd.locate(1, 16); lcd.printf("Robot");

    lcd.locate(2, 1); formatTime(getPlayerTime());
    lcd.locate(2, 16); formatTime(getRobotTime());

    lcd.locate(3, 1); lcd.printf("+%-2d", playerScore);
    lcd.locate(3, 18); lcd.printf("+%-2d", robotScore);

    lcd.locate(4, 1);
    if (statusMessage.length() > 0 && millis() < statusMessageUntil) lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
    else lcd.printf("STOCKFISH     resign  ");
}

void updateDisplay() {
    if (currentMenu == MENU_GAME && gameActive) {
        lcd.locate(2, 1); formatTime(getPlayerTime());
        lcd.locate(2, 16); formatTime(getRobotTime());
        lcd.locate(3, 6); lcd.printf(robotTurn ? "Robot move" : "Your move ");
        lcd.locate(4, 1);
        if (statusMessage.length() > 0 && millis() < statusMessageUntil) lcd.printf("%-20s", shortenMessage(statusMessage).c_str());
        else lcd.printf("STOCKFISH     resign  ");
    }
}

void formatTime(unsigned long milliseconds) {
    unsigned long totalSeconds = milliseconds / 1000;
    unsigned long minutes = totalSeconds / 60;
    unsigned long seconds = totalSeconds % 60;
    lcd.printf("%02lu:%02lu", minutes, seconds);
}
