// ██╗  ██╗███████╗██╗     ██╗      ██████╗   ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
// ██║  ██║██╔════╝██║     ██║     ██╔═══██╗  ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
// ███████║█████╗  ██║     ██║     ██║   ██║  ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
// ██╔══██║██╔══╝  ██║     ██║     ██║   ██║  ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
// ██║  ██║███████╗███████╗███████╗╚██████╔╝  ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
// ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝ ╚═════╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝

// ██████╗  ██████╗ ██████╗  ██████╗ ████████╗     █████╗ ██████╗ ███╗   ███╗
// ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ██╔══██╗██╔══██╗████╗ ████║
// ██████╔╝██║   ██║██████╔╝██║   ██║   ██║       ███████║██████╔╝██╔████╔██║
// ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║       ██╔══██║██╔══██╗██║╚██╔╝██║
// ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║       ██║  ██║██║  ██║██║ ╚═╝ ██║
// ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
// This file needs to read input from chess notation and make the robot arm make that move

// TODO: Initialize variables to the correct pin number
#define SERVO_BASE_PIN     0 // Servo base pin
#define SERVO_ARM_A1_PIN   0 // Servo arm a1 pin
#define SERVO_ARM_A2_PIN   0 // Servo arm a2 pin
#define SERVO_ARM_B_PIN    0 // Servo arm b pin
#define SERVO_WRIST_A_PIN  0 // Servo wrist a pin
#define SERVO_WRIST_B_PIN  0 // Servo wrist b pin
#define SERVO_GRIPPER_PIN  0 // Servo gripper pin

void setup()
{
  Serial.begin(9600);                       // Begin the serial monitor
  pinMode(SERVO_BASE_PIN, OUTPUT);          // Initialize the servo base pin as output
  pinMode(SERVO_ARM_A1_PIN, OUTPUT);        // Initialize the servo arm a1 pin as output
  pinMode(SERVO_ARM_A2_PIN, OUTPUT);        // Initialize the servo arm a2 pin as output
  pinMode(SERVO_ARM_B_PIN, OUTPUT);         // Initialize the servo arm b pin as output
  pinMode(SERVO_WRIST_A_PIN, OUTPUT);       // Initialize the servo wrist a pin as output
  pinMode(SERVO_WRIST_B_PIN, OUTPUT);       // Initialize the servo wrist b pin as output
  pinMode(SERVO_GRIPPER_PIN, OUTPUT);       // Initialize the servo gripper pin as output
}

void loop()
{

}

// Sets the servo's position to a given pulse
void setServo(int servo, unsigned int pulse)
  {
    for (unsigned char i = 0; i < 8; i++)
    {
      digitalWrite(servo, HIGH);
      delayMicroseconds(pulse);
      digitalWrite(servo, LOW);
    }
  }