// NEMA 17 + DRV8825 on Arduino CNC Shield — X, Y, Z axes + SG90 servo
// Servo on pin 12
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

// ─── Servo (raw PWM) ────────────────────────────────────────────────────────

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
// X always counterclockwise (LOW), Z always clockwise (HIGH)

void stepMotorXZ(int steps) {
  digitalWrite(X_DIR, LOW);   // X counterclockwise
  digitalWrite(Z_DIR, HIGH);  // Z clockwise
  delay(1);
  for (int i = 0; i < steps * MICROSTEP; i++) {
    digitalWrite(X_STEP, HIGH);
    digitalWrite(Z_STEP, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(X_STEP, LOW);
    digitalWrite(Z_STEP, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}

// ─── Chess position lookup ────────────────────────────────────────────────────
// Fill in the step counts below for every square on your physical board.
// columSteps  → fed to stepMotorXZ()   (X+Z together, sets arm angle/reach)
// rowSteps    → fed to stepMotor Y     (sets arm height / forward extension)

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
  int colSteps = getColumnSteps(col);
  int rowSteps = getRowSteps(row);

  stepMotorXZ(colSteps);          // X + Z together → column position
  delay(200);
  stepMotor(Y_DIR, Y_STEP, rowSteps, true);  // Y → row position
  delay(200);
}

// ─── Setup ───────────────────────────────────────────────────────────────────

void setup() {
  pinMode(X_DIR,      OUTPUT); pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR,      OUTPUT); pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR,      OUTPUT); pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(SERVO_PIN,  OUTPUT);

  // Keep motors permanently enabled so they hold position/weight
  digitalWrite(ENABLE_PIN, LOW);

  // Center gripper on startup
  setGripper(1500);
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
}