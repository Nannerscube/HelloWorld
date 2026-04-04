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

#define MAX_MOVES 10

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
  goToSquare('A', 1);
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
    delayMicroseconds(3000);
    digitalWrite(X_STEP, LOW);
    delayMicroseconds(3000);
  }
}


void goToA1() {
  rememberMove(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(500);
  rememberMove(X_DIR, X_STEP, 125, true);
  stepMotor(X_DIR, X_STEP, 125, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, true);
  stepMotor(Z_DIR, Z_STEP, 15, true);
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
  delay(2000);
  setGripper(GRIPPER_CLOSED);
  goHome();
}
void goToA2() { /* TODO */ }
void goToA3() { /* TODO */ }
void goToA4() { /* TODO */ }
void goToA5() { /* TODO */ }
void goToA6() { /* TODO */ }
void goToA7() { /* TODO */ }
void goToA8() { /* TODO */ }

void goToB1() { /* TODO */ }
void goToB2() { /* TODO */ }
void goToB3() { /* TODO */ }
void goToB4() { /* TODO */ }
void goToB5() { /* TODO */ }
void goToB6() { /* TODO */ }
void goToB7() { /* TODO */ }
void goToB8() { /* TODO */ }

void goToC1() { /* TODO */ }
void goToC2() { /* TODO */ }
void goToC3() { /* TODO */ }
void goToC4() { /* TODO */ }
void goToC5() { /* TODO */ }
void goToC6() { /* TODO */ }
void goToC7() { /* TODO */ }
void goToC8() { /* TODO */ }

void goToD1() { /* TODO */ }
void goToD2() { /* TODO */ }
void goToD3() { /* TODO */ }
void goToD4() { /* TODO */ }
void goToD5() { /* TODO */ }
void goToD6() { /* TODO */ }
void goToD7() { /* TODO */ }
void goToD8() { /* TODO */ }

void goToE1() { /* TODO */ }
void goToE2() { /* TODO */ }
void goToE3() { /* TODO */ }
void goToE4() { /* TODO */ }
void goToE5() { /* TODO */ }
void goToE6() { /* TODO */ }
void goToE7() { /* TODO */ }
void goToE8() { /* TODO */ }

void goToF1() { /* TODO */ }
void goToF2() { /* TODO */ }
void goToF3() { /* TODO */ }
void goToF4() { /* TODO */ }
void goToF5() { /* TODO */ }
void goToF6() { /* TODO */ }
void goToF7() { /* TODO */ }
void goToF8() { /* TODO */ }

void goToG1() { /* TODO */ }
void goToG2() { /* TODO */ }
void goToG3() { /* TODO */ }
void goToG4() { /* TODO */ }
void goToG5() { /* TODO */ }
void goToG6() { /* TODO */ }
void goToG7() { /* TODO */ }
void goToG8() { /* TODO */ }

void goToH1() { /* TODO */ }
void goToH2() { /* TODO */ }
void goToH3() { /* TODO */ }
void goToH4() { /* TODO */ }
void goToH5() { /* TODO */ }
void goToH6() { /* TODO */ }
void goToH7() { /* TODO */ }
void goToH8() { /* TODO */ }