#include <Arduino_RouterBridge.h>

// Pins
#define ENABLE_PIN 8

#define X_DIR  5
#define X_STEP 2
#define Y_DIR  6
#define Y_STEP 3
#define Z_DIR  7
#define Z_STEP 4

#define Z_ENDSTOP 11

// Values
#define Y_STEP_DELAY_MS 3000
#define Z_STEP_DELAY_MS 8000  // slower than Y
#define Z_EXTRA_STEPS 100     // extra Z after Y stops

#define X_ROTATE_STEPS 168    // steps for 90 degrees more less

void setup() {
  Monitor.begin();

  pinMode(X_DIR, OUTPUT);
  pinMode(X_STEP, OUTPUT);
  pinMode(Y_DIR, OUTPUT);
  pinMode(Y_STEP, OUTPUT);
  pinMode(Z_DIR, OUTPUT);
  pinMode(Z_STEP, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(Z_ENDSTOP, INPUT_PULLUP);

  digitalWrite(ENABLE_PIN, LOW);

  // Start Home
  delay(2000);

  home();
  Monitor.println("Finished YZ Home");
  rotateX90();
  Monitor.println("Finished X 90 Rotation");
}

void loop() {
  // -----
}

void home() {
  
  // Moving Y backward with slow Z lift

  digitalWrite(Y_DIR, LOW); // Y backward
  digitalWrite(Z_DIR, HIGH); // Z up

  int zCounter = 0;

  while (digitalRead(Z_ENDSTOP) == LOW) {  // LOW = pressed | HIGH = not pressed
    // Step Y
    digitalWrite(Y_STEP, HIGH);
    delayMicroseconds(Y_STEP_DELAY_MS);
    digitalWrite(Y_STEP, LOW);
    delayMicroseconds(Y_STEP_DELAY_MS);

    // Step Z slower
    zCounter++;
    if (zCounter >= 3) { // 1 Z step per 3 Y steps
      digitalWrite(Z_STEP, HIGH);
      delayMicroseconds(Z_STEP_DELAY_MS);
      digitalWrite(Z_STEP, LOW);
      delayMicroseconds(Z_STEP_DELAY_MS);
      zCounter = 0;
    }
  }

  // Once Y endstops hits

  // Small back off on Y
  digitalWrite(Y_DIR, HIGH);
  for (int i = 0; i < 20; i++) {
    digitalWrite(Y_STEP, HIGH);
    delayMicroseconds(Y_STEP_DELAY_MS);
    digitalWrite(Y_STEP, LOW);
    delayMicroseconds(Y_STEP_DELAY_MS);
  }

  // Moving Z further up
  for (int i = 0; i < Z_EXTRA_STEPS; i++) {
    digitalWrite(Z_STEP, HIGH);
    delayMicroseconds(Z_STEP_DELAY_MS);
    digitalWrite(Z_STEP, LOW);
    delayMicroseconds(Z_STEP_DELAY_MS);
  }
  
}


void rotateX90() {
  // Rotate 90 degrees for X Home

  digitalWrite(X_DIR, LOW); // Change this to High to change the direction of the rotation
  for (int i = 0; i < X_ROTATE_STEPS; i++) {
    digitalWrite(X_STEP, HIGH);
    delayMicroseconds(3000);
    digitalWrite(X_STEP, LOW);
    delayMicroseconds(3000);
  }

}