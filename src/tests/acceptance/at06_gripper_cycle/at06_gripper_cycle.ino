#define SERVO_PIN 12

#define GRIPPER_CLOSED 2200
#define GRIPPER_OPEN 1000

void setGripper(unsigned int pulse) {
  for (unsigned char i = 0; i < 8; i++) {
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulse);
    digitalWrite(SERVO_PIN, LOW);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(SERVO_PIN, OUTPUT);
  Serial.println("AT-06 Gripper Cycle");
}

void loop() {
  Serial.println("Open gripper");
  setGripper(GRIPPER_OPEN);
  delay(5000);

  Serial.println("Close gripper");
  setGripper(GRIPPER_CLOSED);
  delay(5000);
}
