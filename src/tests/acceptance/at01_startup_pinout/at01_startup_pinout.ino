#define ENABLE_PIN 8

#define X_DIR 5
#define X_STEP 2
#define Y_DIR 6
#define Y_STEP 3
#define Z_DIR 7
#define Z_STEP 4
#define Z_ENDSTOP 11
#define SERVO_PIN 12

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
  pinMode(SERVO_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, LOW);

  Serial.println("AT-01 Startup Pinout");
  Serial.println("Configured outputs: X_DIR=5 X_STEP=2 Y_DIR=6 Y_STEP=3 Z_DIR=7 Z_STEP=4 ENABLE_PIN=8 SERVO_PIN=12");
  Serial.println("Configured input: Z_ENDSTOP=11 INPUT_PULLUP");
  Serial.println("Motor driver enabled with ENABLE_PIN LOW");
}

void loop() {
  delay(1000);
}
