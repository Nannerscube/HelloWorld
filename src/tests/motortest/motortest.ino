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

#define STEPS_PER_REV 200
#define MICROSTEP     1
#define STEP_DELAY_US 1000

#define GRIPPER_CLOSED 2000
#define GRIPPER_OPEN 1000

void servoWrite(int degrees) {
  int pulseUs = map(degrees, 0, 180, SERVO_MIN, SERVO_MAX);
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(pulseUs);
  digitalWrite(SERVO_PIN, LOW);
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

void setup() {
  pinMode(X_DIR,  OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,  OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,  OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  // Center servo on startup
  for (int i = 0; i < 25; i++) {
    servoWrite(90);
    delay(20);
  }
}

void loop() {
  // X
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, true);
  delay(300);
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, false);
  delay(300);

  // Y
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, true);
  delay(300);
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(300);

  // Z
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, true);
  delay(300);
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(300);

  // --- Gripper open ---
  setGripper(GRIPPER_OPEN);
  delay(500);

  // --- Gripper close ---
  setGripper(GRIPPER_CLOSED);
  delay(500);

  delay(500);
}