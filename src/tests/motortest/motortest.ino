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

#define SERVO_MIN 544   // 0 degrees
#define SERVO_MAX 2400  // 180 degrees

#define GRIPPER_CLOSED 2000
#define GRIPPER_OPEN 1000

// --- Servo functions ---
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

// --- Stepper functions ---
void stepMotor(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(STEP_DELAY_US);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(STEP_DELAY_US);
}

// Move X and Z simultaneously
void stepXZ(int steps, bool xCCW, bool zCW) {
  digitalWrite(X_DIR, xCCW ? LOW : HIGH);  // CCW = LOW
  digitalWrite(Z_DIR, zCW  ? HIGH : LOW);  // CW  = HIGH
  delay(1); // allow direction to settle

  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(X_STEP, HIGH);
    digitalWrite(Z_STEP, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(X_STEP, LOW);
    digitalWrite(Z_STEP, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}

// --- Setup ---
void setup() {
  pinMode(X_DIR,  OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,  OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,  OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  // Keep drivers enabled to hold position
  digitalWrite(ENABLE_PIN, LOW);

  // Center servo on startup
  for (int i = 0; i < 25; i++) {
    servoWrite(90);
    delay(20);
  }
}

// --- Main loop ---
void loop() {
  // --- Move X counterclockwise, Z clockwise to same position ---
  stepXZ(STEPS_PER_REV, true, true);

  // Hold position for 2 seconds
  delay(2000);

  // // --- Gripper open ---
  // setGripper(GRIPPER_OPEN);
  // delay(500);

  // // --- Gripper close ---
  // setGripper(GRIPPER_CLOSED);
  // delay(500);

  // --- Move back: X clockwise, Z counterclockwise ---
  stepXZ(STEPS_PER_REV, false, false);

  // Hold position again
  delay(2000);

  // Optional: repeat loop
}