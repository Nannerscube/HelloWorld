#define ENABLE_PIN 8

#define X_DIR 5
#define X_STEP 2
#define Y_DIR 6
#define Y_STEP 3
#define Z_DIR 7
#define Z_STEP 4
#define Z_ENDSTOP 11
#define SERVO_PIN 12

#define STEPS_PER_REV 50
#define MICROSTEP 1
#define STEP_DELAY_US 6000
#define Y_STEP_DELAY_MS 3000
#define Z_STEP_DELAY_MS 8000
#define Z_EXTRA_STEPS 100
#define X_ROTATE_STEPS 168

#define GRIPPER_CLOSED 2200
#define GRIPPER_OPEN 1000

#define MAX_MOVES 30

struct Move {
  int dirPin1, stepPin1;
  bool dir1;
  int dirPin2, stepPin2;
  bool dir2;
  int steps;
  bool isTwoMotors;

  Move() {}
  Move(int dp1, int sp1, bool d1, int dp2, int sp2, bool d2, int s, bool two)
    : dirPin1(dp1), stepPin1(sp1), dir1(d1),
      dirPin2(dp2), stepPin2(sp2), dir2(d2),
      steps(s), isTwoMotors(two) {}
};

Move moveQueue[MAX_MOVES];
int moveCount = 0;

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
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 90);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 90);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 8, false);
  stepMotor(Z_DIR, Z_STEP, 8, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 60);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 60);
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
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 95);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 95);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 18, false);
  stepMotor(Z_DIR, Z_STEP, 18, false);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 45);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 45);
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
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 100);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 100);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 15, true);
  stepMotor(Z_DIR, Z_STEP, 15, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 55);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 55);
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
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 125);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 125);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 70, true);
  stepMotor(Z_DIR, Z_STEP, 70, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 40);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 40);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, 35, true);
  stepMotor(Z_DIR, Z_STEP, 35, true);
  delay(500);
  rememberMove(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 45);
  stepTwoMotors(Z_DIR, Z_STEP, false, Y_DIR, Y_STEP, true, 45);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  goHome();
}

void setup() {
  Serial.begin(115200);

  pinMode(X_DIR, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(Z_ENDSTOP, INPUT_PULLUP);
  pinMode(SERVO_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  delay(2000);
  initializePosition();
  rotateX90(false);
  setGripper(GRIPPER_OPEN);
  Serial.println("AT-08 Square Positions");
}

void loop() {
  Serial.println("Square A1");
  rotateX90(true);
  goToA1();
  delay(1000);

  Serial.println("Square D4");
  rotateX90(true);
  goToD4();
  delay(1000);

  Serial.println("Square E5");
  rotateX90(true);
  goToE5();
  delay(1000);

  Serial.println("Square H8");
  rotateX90(true);
  goToH8();
  delay(1000);

  while (true) {
    delay(1000);
  }
}
