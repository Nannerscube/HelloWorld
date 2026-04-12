#include <Arduino_RouterBridge.h>

#define Z_ENDSTOP 11

void setup() {
  Monitor.begin();

  pinMode(Z_ENDSTOP, INPUT_PULLUP);

  Monitor.println("=== ENDSTOP TEST ===");
}

void loop() {
  int state = digitalRead(Z_ENDSTOP);

  Monitor.print("Endstop state: ");
  Monitor.println(state);

  if (state == HIGH) {
    Monitor.println("NOT PRESSED");
  } else {
    Monitor.println("PRESSED");
  }

  Monitor.println("-------------------");

  delay(500);
}