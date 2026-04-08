#define ENABLE_PIN 8
#define X_DIR 5
#define X_STEP 2

#define X_ROTATE_STEPS 168

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
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  Serial.println("AT-07 X Rotation");
}

void loop() {
  Serial.println("Rotate clockwise");
  rotateX90(true);
  delay(1000);

  Serial.println("Rotate back");
  rotateX90(false);
  delay(1000);

  while (true) {
    delay(1000);
  }
}
