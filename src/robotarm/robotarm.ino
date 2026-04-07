// ██╗  ██╗███████╗██╗     ██╗      ██████╗   ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
// ██║  ██║██╔════╝██║     ██║     ██╔═══██╗  ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
// ███████║█████╗  ██║     ██║     ██║   ██║  ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
// ██╔══██║██╔══╝  ██║     ██║     ██║   ██║  ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
// ██║  ██║███████╗███████╗███████╗╚██████╔╝  ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
// ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝ ╚═════╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝

// ██████╗  ██████╗ ██████╗  ██████╗ ████████╗     █████╗ ██████╗ ███╗   ███╗
// ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ██╔══██╗██╔══██╗████╗ ████║
// ██████╔╝██║   ██║██████╔╝██║   ██║   ██║       ███████║██████╔╝██╔████╔██║
// ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║       ██╔══██║██╔══██╗██║╚██╔╝██║
// ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║       ██║  ██║██║  ██║██║ ╚═╝ ██║
// ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
// This file needs to read input from chess notation and make the robot arm make that 

#include <Arduino_RouterBridge.h>


// ██████╗ ██╗███╗   ██╗ ██████╗ ██╗   ██╗████████╗
// ██╔══██╗██║████╗  ██║██╔═══██╗██║   ██║╚══██╔══╝
// ██████╔╝██║██╔██╗ ██║██║   ██║██║   ██║   ██║   
// ██╔═══╝ ██║██║╚██╗██║██║   ██║██║   ██║   ██║   
// ██║     ██║██║ ╚████║╚██████╔╝╚██████╔╝   ██║   
// ╚═╝     ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝    ╚═╝   

#define ENABLE_PIN 8

#define X_DIR  5
#define X_STEP 2

#define Y_DIR  6
#define Y_STEP 3

#define Z_DIR  7
#define Z_STEP 4
#define Z_ENDSTOP 11

#define SERVO_PIN 12


//  ██████╗ ██████╗ ███╗   ██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗███████╗
// ██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝██╔════╝
// ██║     ██║   ██║██╔██╗ ██║███████╗   ██║   ███████║██╔██╗ ██║   ██║   ███████╗
// ██║     ██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║   ╚════██║
// ╚██████╗╚██████╔╝██║ ╚████║███████║   ██║   ██║  ██║██║ ╚████║   ██║   ███████║
//  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝

#define STEPS_PER_REV 50
#define MICROSTEP     1
#define STEP_DELAY_US 6000
#define Y_STEP_DELAY_MS 3000
#define Z_STEP_DELAY_MS 8000  // slower than Y
#define Z_EXTRA_STEPS 100     // extra Z after Y stops

#define X_ROTATE_STEPS 168    // steps for 90 degrees more less

#define GRIPPER_CLOSED 2200
#define GRIPPER_OPEN   1000

#define MAX_MOVES 30

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


// ███████╗███████╗████████╗██╗   ██╗██████╗ 
// ██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗
// ███████╗█████╗     ██║   ██║   ██║██████╔╝
// ╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝ 
// ███████║███████╗   ██║   ╚██████╔╝██║     
// ╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     

void setup() {
  Monitor.begin();

  pinMode(X_DIR, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(Z_ENDSTOP, INPUT_PULLUP);

  digitalWrite(ENABLE_PIN, LOW);

  // Go to initial position
  delay(2000);
  initializePosition();
  Monitor.println("Finished YZ Home");
  rotateX90(false);
  Monitor.println("Finished X 90 Rotation");
  delay(500);
}


// ██╗      ██████╗  ██████╗ ██████╗ 
// ██║     ██╔═══██╗██╔═══██╗██╔══██╗
// ██║     ██║   ██║██║   ██║██████╔╝
// ██║     ██║   ██║██║   ██║██╔═══╝ 
// ███████╗╚██████╔╝╚██████╔╝██║     
// ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     

void loop() {
  setGripper(GRIPPER_OPEN);
  goToSquare('h', 8);
  while(true);
}


// ██╗      ██████╗  ██████╗ ██╗ ██████╗
// ██║     ██╔═══██╗██╔════╝ ██║██╔════╝
// ██║     ██║   ██║██║  ███╗██║██║     
// ██║     ██║   ██║██║   ██║██║██║     
// ███████╗╚██████╔╝╚██████╔╝██║╚██████╗
// ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝
// Takes a square in chess notation (e.g. "A1", "H8") and moves the arm to it
void goToSquare(char col, int row) {
  moveCount = 0;
  rotateX90(true);
  switch (col) {
    case 'A': case 'a':
      switch (row) {
        case 1: goToA1(); break;
        case 2: goToA2(); break;
        case 3: goToA3(); break;
        case 4: goToA4(); break;
        case 5: goToA5(); break;
        case 6: goToA6(); break;
        case 7: goToA7(); break;
        case 8: goToA8(); break;
      } break;

    case 'B': case 'b':
      switch (row) {
        case 1: goToB1(); break;
        case 2: goToB2(); break;
        case 3: goToB3(); break;
        case 4: goToB4(); break;
        case 5: goToB5(); break;
        case 6: goToB6(); break;
        case 7: goToB7(); break;
        case 8: goToB8(); break;
      } break;

    case 'C': case 'c':
      switch (row) {
        case 1: goToC1(); break;
        case 2: goToC2(); break;
        case 3: goToC3(); break;
        case 4: goToC4(); break;
        case 5: goToC5(); break;
        case 6: goToC6(); break;
        case 7: goToC7(); break;
        case 8: goToC8(); break;
      } break;

    case 'D': case 'd':
      switch (row) {
        case 1: goToD1(); break;
        case 2: goToD2(); break;
        case 3: goToD3(); break;
        case 4: goToD4(); break;
        case 5: goToD5(); break;
        case 6: goToD6(); break;
        case 7: goToD7(); break;
        case 8: goToD8(); break;
      } break;

    case 'E': case 'e':
      switch (row) {
        case 1: goToE1(); break;
        case 2: goToE2(); break;
        case 3: goToE3(); break;
        case 4: goToE4(); break;
        case 5: goToE5(); break;
        case 6: goToE6(); break;
        case 7: goToE7(); break;
        case 8: goToE8(); break;
      } break;

    case 'F': case 'f':
      switch (row) {
        case 1: goToF1(); break;
        case 2: goToF2(); break;
        case 3: goToF3(); break;
        case 4: goToF4(); break;
        case 5: goToF5(); break;
        case 6: goToF6(); break;
        case 7: goToF7(); break;
        case 8: goToF8(); break;
      } break;

    case 'G': case 'g':
      switch (row) {
        case 1: goToG1(); break;
        case 2: goToG2(); break;
        case 3: goToG3(); break;
        case 4: goToG4(); break;
        case 5: goToG5(); break;
        case 6: goToG6(); break;
        case 7: goToG7(); break;
        case 8: goToG8(); break;
      } break;

    case 'H': case 'h':
      switch (row) {
        case 1: goToH1(); break;
        case 2: goToH2(); break;
        case 3: goToH3(); break;
        case 4: goToH4(); break;
        case 5: goToH5(); break;
        case 6: goToH6(); break;
        case 7: goToH7(); break;
        case 8: goToH8(); break;
      } break;
  }
}


// ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
// ██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
// ██████╔╝█████╗  ██║     ██║   ██║██████╔╝██║  ██║██║██╔██╗ ██║██║  ███╗
// ██╔══██╗██╔══╝  ██║     ██║   ██║██╔══██╗██║  ██║██║██║╚██╗██║██║   ██║
// ██║  ██║███████╗╚██████╗╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
// ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 

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


// ███╗   ███╗ ██████╗ ████████╗ ██████╗ ██████╗ ███████╗
// ████╗ ████║██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
// ██╔████╔██║██║   ██║   ██║   ██║   ██║██████╔╝███████╗
// ██║╚██╔╝██║██║   ██║   ██║   ██║   ██║██╔══██╗╚════██║
// ██║ ╚═╝ ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║███████║
// ╚═╝     ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝

void setGripper(unsigned int pulse) {
  for (unsigned char i = 0; i < 8; i++) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulse);
    digitalWrite(SERVO_PIN, LOW);
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


// ██████╗  ██████╗ ███████╗██╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
// ██╔══██╗██╔═══██╗██╔════╝██║╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
// ██████╔╝██║   ██║███████╗██║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
// ██╔═══╝ ██║   ██║╚════██║██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
// ██║     ╚██████╔╝███████║██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
// ╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

void initializePosition() {
  digitalWrite(Y_DIR, LOW);  // Y backward
  digitalWrite(Z_DIR, HIGH); // Z up

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

  // Small back off on Y
  digitalWrite(Y_DIR, HIGH);
  for (int i = 0; i < 20; i++) {
    digitalWrite(Y_STEP, HIGH);
    delayMicroseconds(Y_STEP_DELAY_MS);
    digitalWrite(Y_STEP, LOW);
    delayMicroseconds(Y_STEP_DELAY_MS);
  }

  // Moving Z further up
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


void goToA1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 120, true);
  stepMotor(X_DIR, X_STEP, 120, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 104  , true);
  stepMotor(X_DIR, X_STEP, 104, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                107);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                107);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 91, true);
  stepMotor(X_DIR, X_STEP, 91, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 55, true);
  stepMotor(Z_DIR, Z_STEP, 55, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                107);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                107);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 79, true);
  stepMotor(X_DIR, X_STEP, 79, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 75, true);
  stepMotor(Z_DIR, Z_STEP, 75, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 71, true);
  stepMotor(X_DIR, X_STEP, 71, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 40, true);
  stepMotor(Z_DIR, Z_STEP, 40, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA6() {
    rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 70, true);
  stepMotor(X_DIR, X_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                40);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                40);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 65, true);
  stepMotor(X_DIR, X_STEP, 65, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 95, true);
  stepMotor(Z_DIR, Z_STEP, 95, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                55);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 58, true);
  stepMotor(X_DIR, X_STEP, 58, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                33);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                33);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                60);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                60);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToB1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 94, true);
  stepMotor(X_DIR, X_STEP, 94, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 40, false);
  stepMotor(Z_DIR, Z_STEP, 40, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 80, true);
  stepMotor(X_DIR, X_STEP, 80, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 66, true);
  stepMotor(X_DIR, X_STEP, 66, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 57, true);
  stepMotor(X_DIR, X_STEP, 57, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                105);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                105);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 58, true);
  stepMotor(X_DIR, X_STEP, 58, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                27);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                27);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 53, true);
  stepMotor(X_DIR, X_STEP, 53, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 90, true);
  stepMotor(Z_DIR, Z_STEP, 90, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 40, true);
  stepMotor(Z_DIR, Z_STEP, 40, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 44, true);
  stepMotor(X_DIR, X_STEP, 44, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                80);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                80);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                15);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                15);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToB8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 42, true);
  stepMotor(X_DIR, X_STEP, 42, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToC1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 70, true);
  stepMotor(X_DIR, X_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, false);
  stepMotor(Z_DIR, Z_STEP, 35, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 55, true);
  stepMotor(X_DIR, X_STEP, 55, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                85);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                85);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, false);
  stepMotor(Z_DIR, Z_STEP, 30, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                40);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                40);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 43, true);
  stepMotor(X_DIR, X_STEP, 43, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 45, true);
  stepMotor(Z_DIR, Z_STEP, 45, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                100);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                100);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 22, false);
  stepMotor(Z_DIR, Z_STEP, 22, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 38, true);
  stepMotor(X_DIR, X_STEP, 38, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 65, true);
  stepMotor(Z_DIR, Z_STEP, 65, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 35, true);
  stepMotor(X_DIR, X_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 12, true);
  stepMotor(Z_DIR, Z_STEP, 12, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                20);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 32, true);
  stepMotor(X_DIR, X_STEP, 32, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 30, true);
  stepMotor(X_DIR, X_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 90, true);
  stepMotor(Z_DIR, Z_STEP, 90, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToC8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 30, true);
  stepMotor(X_DIR, X_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                80);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                80);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                30);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                10);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToD1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 35, true);
  stepMotor(X_DIR, X_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                60);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                60);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 45, false);
  stepMotor(Z_DIR, Z_STEP, 45, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 22, true);
  stepMotor(X_DIR, X_STEP, 22, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, false);
  stepMotor(Z_DIR, Z_STEP, 35, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                55);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 18, true);
  stepMotor(X_DIR, X_STEP, 18, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 40, true);
  stepMotor(Z_DIR, Z_STEP, 40, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                90);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, false);
  stepMotor(Z_DIR, Z_STEP, 20, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 15, true);
  stepMotor(X_DIR, X_STEP, 15, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 18, false);
  stepMotor(Z_DIR, Z_STEP, 18, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 16, true);
  stepMotor(X_DIR, X_STEP, 16, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               28);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               28);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 14, true);
  stepMotor(X_DIR, X_STEP, 14, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 85, true);
  stepMotor(Z_DIR, Z_STEP, 85, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 15, true);
  stepMotor(X_DIR, X_STEP, 15, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 95, true);
  stepMotor(Z_DIR, Z_STEP, 95, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 45, true);
  stepMotor(Z_DIR, Z_STEP, 45, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToD8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 14, true);
  stepMotor(X_DIR, X_STEP, 14, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               60);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               60);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToE1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 20, false);
  stepMotor(X_DIR, X_STEP, 20, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 45, false);
  stepMotor(Z_DIR, Z_STEP, 45, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 15, false);
  stepMotor(X_DIR, X_STEP, 15, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, false);
  stepMotor(Z_DIR, Z_STEP, 35, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 10, false);
  stepMotor(X_DIR, X_STEP, 10, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                85);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                85);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, false);
  stepMotor(Z_DIR, Z_STEP, 20, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 5, false);
  stepMotor(X_DIR, X_STEP, 5, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                105);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                105);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 65, true);
  stepMotor(Z_DIR, Z_STEP, 65, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                115);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, true);
  stepMotor(Z_DIR, Z_STEP, 15, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                33);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                33);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 3, false);
  stepMotor(X_DIR, X_STEP, 3, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 80, true);
  stepMotor(Z_DIR, Z_STEP, 80, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                120);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                120);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                35);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 3, false);
  stepMotor(X_DIR, X_STEP, 3, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 90, true);
  stepMotor(Z_DIR, Z_STEP, 90, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 45, true);
  stepMotor(Z_DIR, Z_STEP, 45, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                45);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToE8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 3, false);
  stepMotor(X_DIR, X_STEP, 3, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                15);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                15);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, true);
  stepMotor(Z_DIR, Z_STEP, 15, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                52);
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                52);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToF1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 55, false);
  stepMotor(X_DIR, X_STEP, 55, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 30, false);
  stepMotor(Z_DIR, Z_STEP, 30, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 45, false);
  stepMotor(X_DIR, X_STEP, 45, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, false);
  stepMotor(Z_DIR, Z_STEP, 35, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 33, false);
  stepMotor(X_DIR, X_STEP, 33, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 45, true);
  stepMotor(Z_DIR, Z_STEP, 45, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               85);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               85);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 20, false);
  stepMotor(Z_DIR, Z_STEP, 20, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 25, false);
  stepMotor(X_DIR, X_STEP, 25, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               100);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               100);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, false);
  stepMotor(Z_DIR, Z_STEP, 15, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 22, false);
  stepMotor(X_DIR, X_STEP, 22, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               115);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               115);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 12, true);
  stepMotor(Z_DIR, Z_STEP, 12, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 15, false);
  stepMotor(X_DIR, X_STEP, 15, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 85, true);
  stepMotor(Z_DIR, Z_STEP, 85, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 12, false);
  stepMotor(X_DIR, X_STEP, 12, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 95, true);
  stepMotor(Z_DIR, Z_STEP, 95, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               50);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToF8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 14, false);
  stepMotor(X_DIR, X_STEP, 14, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToG1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 88, false);
  stepMotor(X_DIR, X_STEP, 88, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, false);
  stepMotor(Z_DIR, Z_STEP, 15, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               63);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               63);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 70, false);
  stepMotor(X_DIR, X_STEP, 70, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 10, false);
  stepMotor(Z_DIR, Z_STEP, 10, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 55, false);
  stepMotor(X_DIR, X_STEP, 55, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 3, true);
  stepMotor(Z_DIR, Z_STEP, 3, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 50, false);
  stepMotor(X_DIR, X_STEP, 50, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               95);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               95);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 10, true);
  stepMotor(Z_DIR, Z_STEP, 10, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 40, false);
  stepMotor(X_DIR, X_STEP, 40, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 65, true);
  stepMotor(Z_DIR, Z_STEP, 65, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 20, true);
  stepMotor(Z_DIR, Z_STEP, 20, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 36, false);
  stepMotor(X_DIR, X_STEP, 36, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 80, true);
  stepMotor(Z_DIR, Z_STEP, 80, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToG7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 35, false);
  stepMotor(X_DIR, X_STEP, 35, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 90, true);
  stepMotor(Z_DIR, Z_STEP, 90, true);
  delay(500);
  
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               80);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               80);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 30, true);
  stepMotor(Z_DIR, Z_STEP, 30, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               10);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               10);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToG8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 28, false);
  stepMotor(X_DIR, X_STEP, 28, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 90, true);
  stepMotor(Z_DIR, Z_STEP, 90, true);
  delay(500);
  
  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               80);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               80);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               75);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               30);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}

void goToH1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 110, false);
  stepMotor(X_DIR, X_STEP, 110, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               65);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 10, true);
  stepMotor(Z_DIR, Z_STEP, 10, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               63);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               63);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH2() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 90, false);
  stepMotor(X_DIR, X_STEP, 90, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               70);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 5, true);
  stepMotor(Z_DIR, Z_STEP, 5, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               55);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH3() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 75, false);
  stepMotor(X_DIR, X_STEP, 75, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               85);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               85);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 10, true);
  stepMotor(Z_DIR, Z_STEP, 10, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH4() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 60, false);
  stepMotor(X_DIR, X_STEP, 60, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 50, true);
  stepMotor(Z_DIR, Z_STEP, 50, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               95);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               95);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 25, true);
  stepMotor(Z_DIR, Z_STEP, 25, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH5() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 55, false);
  stepMotor(X_DIR, X_STEP, 55, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 65, true);
  stepMotor(Z_DIR, Z_STEP, 65, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               110);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 40, true);
  stepMotor(Z_DIR, Z_STEP, 40, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH6() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 45, false);
  stepMotor(X_DIR, X_STEP, 45, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 80, true);
  stepMotor(Z_DIR, Z_STEP, 80, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               120);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 45, true);
  stepMotor(Z_DIR, Z_STEP, 45, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               35);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH7() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 43, false);
  stepMotor(X_DIR, X_STEP, 43, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 95, true);
  stepMotor(Z_DIR, Z_STEP, 95, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 60, true);
  stepMotor(Z_DIR, Z_STEP, 60, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               50);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               50);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToH8() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);

  rememberMove(X_DIR, X_STEP, 42, false);
  stepMotor(X_DIR, X_STEP, 42, false);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 100, true);
  stepMotor(Z_DIR, Z_STEP, 100, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               125);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               40);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);

  rememberMove(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  stepTwoMotors(Z_DIR, Z_STEP, false,
               Y_DIR, Y_STEP, true,
               45);
  delay(500);

  setGripper(GRIPPER_CLOSED);
  goHome();
}