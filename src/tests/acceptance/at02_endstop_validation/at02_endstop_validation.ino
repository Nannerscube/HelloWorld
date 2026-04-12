#define Z_ENDSTOP 11

void setup() {
  Serial.begin(115200);
  pinMode(Z_ENDSTOP, INPUT_PULLUP);
  Serial.println("AT-02 Endstop Validation");
}

void loop() {
  int state = digitalRead(Z_ENDSTOP);

  Serial.print("Endstop state: ");
  Serial.println(state);

  if (state == HIGH) {
    Serial.println("NOT PRESSED");
  } else {
    Serial.println("PRESSED");
  }

  Serial.println("---");
  delay(500);
}
