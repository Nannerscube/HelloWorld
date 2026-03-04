#include <Servo.h>  // needs to be in arduino ide
Servo MyServo;
void setup()
{
MyServo.attach(9);     // White - wrist pivot
MyServo.attach(10);    // Green - base pivot
MyServo.attach(11);    // Purple - elbow
MyServo.attach(6);     // Brown - gripper
MyServo.attach(5);     // Blue  - base rotate
MyServo.attach(3);     // Grey - wrist rotate
}
void loop()
{
MyServo.write(90); // Go to 0 degrees
delay(5000); // Wait 5 seconds
MyServo.write(0); // Go to 90 degrees
delay(5000); // Wait 5 seconds
MyServo.write(180); // Go to 180 degrees
delay(10000); // Wait 10 seconds
}