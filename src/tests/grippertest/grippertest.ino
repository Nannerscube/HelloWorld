#define SERVO_PIN 12

// Adjust
#define SERVO_MIN 1000   // open
#define SERVO_MAX 2500   // closed

void sendPulse(int pulse) {
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(pulse);
  digitalWrite(SERVO_PIN, LOW);
  delay(20); // 20ms period
}

void setup() {
  pinMode(SERVO_PIN, OUTPUT);

  // fully closed position
  for (int i = 0; i < 50; i++) {   // 1s
    sendPulse(SERVO_MAX);
  }
}

void loop() {
  // keep closed
  sendPulse(SERVO_MAX);
  delay(5000);
  sendPulse(SERVO_MIN);
  delay(5000);
}