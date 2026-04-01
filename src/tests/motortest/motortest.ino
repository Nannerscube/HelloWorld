// NEMA 17 + DRV8825 on Arduino CNC Shield — X, Y, Z axes + SG90 servo
// Servo on pin 12 via raw PWM (no Servo library — works on Arduino UNO Q)

#define ENABLE_PIN 8

#define X_DIR  5
#define X_STEP 2

#define Y_DIR  6
#define Y_STEP 3

#define Z_DIR  7
#define Z_STEP 4

#define SERVO_PIN 12

#define STEPS_PER_REV 50
#define MICROSTEP     1
#define STEP_DELAY_US 6000

#define GRIPPER_CLOSED 2200
#define GRIPPER_OPEN 1000

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

void setup() {
  pinMode(X_DIR,  OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,  OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,  OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);
}

void loop() {
  // ── FORWARD SEQUENCE ──────────────────────────────────────────────

  // Open gripper
  setGripper(GRIPPER_OPEN);
  delay(500);

  // Move 1: Z up + Y forward
  stepTwoMotors(Z_DIR, Z_STEP, true,
                Y_DIR, Y_STEP, true,
                STEPS_PER_REV);
  delay(300);

  // Move 2: X forward
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, true);
  delay(500);

  // Move 3: X back
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, false);
  delay(500);

  // Move 4: Z up + Y forward again
  stepTwoMotors(Z_DIR, Z_STEP, true,
                Y_DIR, Y_STEP, true,
                STEPS_PER_REV);
  delay(300);

  // Move 5: Z down + Y forward (60 steps), then close gripper
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                50);
  delay(500);
  setGripper(GRIPPER_CLOSED);
  delay(500);

  // ── RETURN SEQUENCE (reversed) ────────────────────────────────────

  // Undo Move 5: Z up + Y back (60 steps)
  stepTwoMotors(Z_DIR, Z_STEP, true,
                Y_DIR, Y_STEP, false,
                50);
  delay(300);

  // Undo Move 4: Z down + Y back
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, false,
                STEPS_PER_REV);
  delay(300);

  // Undo Move 3: X forward (was already undone by Move 3 itself — X went forward then back)
  // Net X displacement = 0, no undo needed

  // Undo Move 1: Z down + Y back
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, false,
                STEPS_PER_REV);
  delay(300);

  // Re-open gripper back to resting state
  setGripper(GRIPPER_OPEN);
  delay(500);

  while (true);
}