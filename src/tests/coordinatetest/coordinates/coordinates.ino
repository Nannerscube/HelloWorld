// NEMA 17 + DRV8825 on Arduino CNC Shield — X, Y, Z axes + SG90 servo
// Servo on pin 12 via raw PWM (no Servo library)
// X = forward/back, Y = left/right, Z = up/down

#define ENABLE_PIN 8

#define X_DIR  5
#define X_STEP 2

#define Y_DIR  6
#define Y_STEP 3

#define Z_DIR  7
#define Z_STEP 4

#define SERVO_PIN 12

#define STEPS_PER_REV   200
#define MICROSTEP       1
#define STEP_DELAY_US   1000

#define GRIPPER_CLOSED  2000
#define GRIPPER_OPEN    1000

// ─── Home coordinates ────────────────────────────────────────────────────────
#define HOME_X  0    // TODO: set to your home X step position
#define HOME_Y  0    // TODO: set to your home Y step position
#define HOME_Z  0    // TODO: set to your home Z step position

// ─── Position tracking ───────────────────────────────────────────────────────
int posX = 0;
int posY = 0;
int posZ = 0;

// ─── Servo (raw PWM) ─────────────────────────────────────────────────────────
void setGripper(unsigned int pulse) {
  for (unsigned char i = 0; i < 8; i++) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulse);
    digitalWrite(SERVO_PIN, LOW);
    delay(20);
  }
}

// ─── Single-axis step ────────────────────────────────────────────────────────
void stepMotor(int dirPin, int stepPin, int steps, bool clockwise) {
  if (steps <= 0) return;
  digitalWrite(dirPin, clockwise ? HIGH : LOW);
  delay(1);
  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}

// ─── Move to absolute step position ─────────────────────────────────────────
// X = forward/back, Y = left/right, Z = up/down
// Moves Z first (lift), then X/Y (travel), then Z down (lower)

void moveToPosition(int targetX, int targetY, int targetZ) {
  // Lift Z before moving (avoids dragging along the board)
  int deltaZ = targetZ - posZ;
  if (deltaZ != 0) {
    stepMotor(Z_DIR, Z_STEP, abs(deltaZ), deltaZ > 0);
    posZ += deltaZ;
  }
  delay(200);

  // Move X (forward/back)
  int deltaX = targetX - posX;
  if (deltaX != 0) {
    stepMotor(X_DIR, X_STEP, abs(deltaX), deltaX > 0);
    posX += deltaX;
  }
  delay(200);

  // Move Y (left/right)
  int deltaY = targetY - posY;
  if (deltaY != 0) {
    stepMotor(Y_DIR, Y_STEP, abs(deltaY), deltaY > 0);
    posY += deltaY;
  }
  delay(200);
}

// ─── Go to home position ─────────────────────────────────────────────────────
void goHome() {
  moveToPosition(0, 0, 0);
}

// ─── Startup homing sequence ─────────────────────────────────────────────────
void driveToHomeOnStartup() {
  if (HOME_Z != 0) {
    stepMotor(Z_DIR, Z_STEP, abs(HOME_Z), HOME_Z > 0);
  }
  delay(300);

  if (HOME_X != 0) {
    stepMotor(X_DIR, X_STEP, abs(HOME_X), HOME_X > 0);
  }
  delay(300);

  if (HOME_Y != 0) {
    stepMotor(Y_DIR, Y_STEP, abs(HOME_Y), HOME_Y > 0);
  }
  delay(300);

  posX = 0;
  posY = 0;
  posZ = 0;
}

// ─── Setup ───────────────────────────────────────────────────────────────────
void setup() {
  pinMode(X_DIR,      OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,      OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,      OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  setGripper(GRIPPER_OPEN);
  delay(300);

  driveToHomeOnStartup();
}

// ─── Loop ────────────────────────────────────────────────────────────────────
// Edit these values to move the arm wherever you need.
// X = forward/back steps, Y = left/right steps, Z = up/down steps

void loop() {
  moveToPosition(100, 0, 100);   // move to target
  delay(500);

  moveToPosition(-100, 0, -100);  // move to destination
  delay(500);

}