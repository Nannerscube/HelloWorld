#define ENABLE_PIN 8

#define Y_DIR 6
#define Y_STEP 3
#define Z_DIR 7
#define Z_STEP 4

#define MICROSTEP 1
#define STEP_DELAY_US 6000

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
  Serial.begin(115200);

  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  Serial.println("AT-05 Combined YZ Motion");
}

void loop() {
  Serial.println("Forward combined move");
  stepTwoMotors(Z_DIR, Z_STEP, false,
                Y_DIR, Y_STEP, true,
                75);
  delay(500);

  Serial.println("Reverse combined move");
  stepTwoMotors(Z_DIR, Z_STEP, true,
                Y_DIR, Y_STEP, false,
                75);
  delay(1000);

  while (true) {
    delay(1000);
  }
}
