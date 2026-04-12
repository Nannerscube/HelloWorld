#include <Bridge.h>  // UNO Q bridge library

void setup() {
    Bridge.begin();  // start bridge
}

void loop() {
    Bridge.delay(10);
}

// this is the function pytohn sees
void fromC() {
    Serial.println("from C");
}
