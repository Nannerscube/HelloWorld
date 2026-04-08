#define ENABLE_PIN 8

#define X_DIR 5
#define X_STEP 2
#define Y_DIR 6
#define Y_STEP 3
#define Z_DIR 7
#define Z_STEP 4
#define Z_ENDSTOP 11

#define Y_STEP_DELAY_MS 3000
#define Z_STEP_DELAY_MS 8000
#define Z_EXTRA_STEPS 100
#define X_ROTATE_STEPS 168

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

  digitalWrite(ENABLE_PIN, LOW);

  Serial.println("AT-03 Homing Cycle");
  delay(2000);
  initializePosition();
  Serial.println("Homing complete");
  rotateX90(false);
  Serial.println("Returned to startup X orientation");
}

void loop() {
  delay(1000);
}
