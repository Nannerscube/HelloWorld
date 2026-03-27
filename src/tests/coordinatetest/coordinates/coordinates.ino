// NEMA 17 + DRV8825 on Arduino CNC Shield — X, Y, Z axes + SG90 servo
// Servo on pin 12 via raw PWM (no Servo library)
// Chess arm: X+Z move together (column), Y moves independently (row)

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
// Set these to the step counts that position the arm at your desired home.
// The arm will drive to these on startup and treat them as origin (0,0).

#define HOME_XZ   0    // TODO: set to your home XZ step position
#define HOME_Y    0    // TODO: set to your home Y step position

// ─── Position tracking ───────────────────────────────────────────────────────

int posXZ = 0;
int posY  = 0;

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

// ─── X + Z simultaneous ──────────────────────────────────────────────────────
// forward=true  → X counterclockwise, Z clockwise   (away from home)
// forward=false → X clockwise,        Z counterclockwise (toward home)

void stepMotorXZ(int steps, bool forward) {
  if (steps <= 0) return;
  digitalWrite(X_DIR, forward ? LOW  : HIGH);
  digitalWrite(Z_DIR, forward ? HIGH : LOW);
  delay(1);
  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(X_STEP, HIGH);
    digitalWrite(Z_STEP, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(X_STEP, LOW);
    digitalWrite(Z_STEP, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
  posXZ += forward ? steps : -steps;
}

// ─── Move to absolute step position ─────────────────────────────────────────
// Drives each axis to an exact step count from home, regardless of where
// the arm currently is.

void moveToPosition(int targetXZ, int targetY) {
  // ── XZ axis ──
  int deltaXZ = targetXZ - posXZ;
  if (deltaXZ != 0) {
    stepMotorXZ(abs(deltaXZ), deltaXZ > 0);
  }

  delay(200);

  // ── Y axis ──
  int deltaY = targetY - posY;
  if (deltaY != 0) {
    bool clockwise = (deltaY > 0);
    stepMotor(Y_DIR, Y_STEP, abs(deltaY), clockwise);
    posY += deltaY;
  }

  delay(200);
}

// ─── Go to home position ─────────────────────────────────────────────────────

void goHome() {
  moveToPosition(0, 0);
}

// ─── Chess position lookup ────────────────────────────────────────────────────

int getColumnSteps(char col) {
  switch (col) {
    case 'A': return 0;    // TODO: calibrate
    case 'B': return 25;
    case 'C': return 50;
    case 'D': return 75;
    case 'E': return 100;
    case 'F': return 125;
    case 'G': return 150;
    case 'H': return 175;
    default:  return 0;
  }
}

int getRowSteps(int row) {
  switch (row) {
    case 1: return 0;    // TODO: calibrate
    case 2: return 25;
    case 3: return 50;
    case 4: return 75;
    case 5: return 100;
    case 6: return 125;
    case 7: return 150;
    case 8: return 175;
    default: return 0;
  }
}

// ─── Move arm to a chess square ──────────────────────────────────────────────

void goToChessSquare(char col, int row) {
  goHome();
  delay(300);
  moveToPosition(getColumnSteps(col), getRowSteps(row));
}

// ─── Startup homing sequence ──────────────────────────────────────────────────
// Drives the arm from its physical power-on position to HOME_XZ / HOME_Y,
// then resets the position counters to 0 so that (0,0) = home from here on.
//
// HOW TO SET HOME_XZ AND HOME_Y:
//   1. Place the arm at the furthest position it could ever start from.
//   2. Count how many steps it takes in each axis to reach your desired home.
//   3. Enter those counts as HOME_XZ and HOME_Y above.
//   4. Every power-on the arm will drive those steps and stop there.

void driveToHomeOnStartup() {
  // Drive XZ to home
  if (HOME_XZ > 0) {
    stepMotorXZ(HOME_XZ, true);     // move forward to home
  } else if (HOME_XZ < 0) {
    stepMotorXZ(-HOME_XZ, false);   // move backward to home
  }
  delay(300);

  // Drive Y to home
  if (HOME_Y != 0) {
    bool clockwise = (HOME_Y > 0);
    stepMotor(Y_DIR, Y_STEP, abs(HOME_Y), clockwise);
  }
  delay(300);

  // Declare this as position zero — all future moves are relative to here
  posXZ = 0;
  posY  = 0;
}

// ─── Setup ───────────────────────────────────────────────────────────────────

void setup() {
  pinMode(X_DIR,      OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,      OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,      OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  // Motors always on — hold position and weight at all times
  digitalWrite(ENABLE_PIN, LOW);

  // Open gripper on startup
  setGripper(GRIPPER_OPEN);
  delay(300);

  // Drive to home and lock in (0,0) as origin
  driveToHomeOnStartup();
}

// ─── Loop (example sequence) ─────────────────────────────────────────────────

void loop() {
  // Example: pick up piece at D4, drop at D5
  goToChessSquare('D', 4);
  setGripper(GRIPPER_CLOSED);
  delay(500);

  goToChessSquare('D', 5);
  setGripper(GRIPPER_OPEN);
  delay(500);

  goHome();
  delay(2000);
}