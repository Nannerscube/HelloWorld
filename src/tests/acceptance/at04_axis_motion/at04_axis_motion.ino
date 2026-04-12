#define ENABLE_PIN 8

#define X_DIR 5
#define X_STEP 2
#define Y_DIR 6
#define Y_STEP 3
#define Z_DIR 7
#define Z_STEP 4

#define STEPS_PER_REV 50
#define MICROSTEP 1
#define STEP_DELAY_US 6000

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
  Serial.begin(115200);

  pinMode(X_DIR, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  Serial.println("AT-04 Axis Motion");
}

void loop() {
  Serial.println("X forward");
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, true);
  delay(500);

  Serial.println("X reverse");
  stepMotor(X_DIR, X_STEP, STEPS_PER_REV, false);
  delay(500);

  Serial.println("Y forward");
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, true);
  delay(500);

  Serial.println("Y reverse");
  stepMotor(Y_DIR, Y_STEP, STEPS_PER_REV, false);
  delay(500);

  Serial.println("Z up");
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, true);
  delay(500);

  Serial.println("Z down");
  stepMotor(Z_DIR, Z_STEP, STEPS_PER_REV, false);
  delay(1000);

  while (true) {
    delay(1000);
  }
}
