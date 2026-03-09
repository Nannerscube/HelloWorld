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
#define X_STEP_PIN         0
#define X_DIR_PIN          0
#define X_ENABLE_PIN       0
#define X_MIN_PIN          0
#define X_MAX_PIN          0

#define Y_STEP_PIN         0
#define Y_DIR_PIN          0
#define Y_ENABLE_PIN       0
#define Y_MIN_PIN          0
#define Y_MAX_PIN          0

#define Z_STEP_PIN         0
#define Z_DIR_PIN          0
#define Z_ENABLE_PIN       0
#define Z_MIN_PIN          0
#define Z_MAX_PIN          0

#define E_STEP_PIN         0
#define E_DIR_PIN          0
#define E_ENABLE_PIN       0

#define Q_STEP_PIN         0
#define Q_DIR_PIN          0
#define Q_ENABLE_PIN       0

#define SDPOWER            -1
#define SDSS               0
#define LED_PIN            0

#define FAN_PIN            0

#define PS_ON_PIN          0
#define KILL_PIN           -1

#define HEATER_0_PIN       0
#define HEATER_1_PIN       0
#define TEMP_0_PIN         0   // ANALOG NUMBERING
#define TEMP_1_PIN         0   // ANALOG NUMBERING

#define STEPPER_GRIPPER_PIN_0 0
#define STEPPER_GRIPPER_PIN_1 0
#define STEPPER_GRIPPER_PIN_2 0
#define STEPPER_GRIPPER_PIN_3 0

class RobotGeometry {
public:
  RobotGeometry();
  void set(float axmm, float aymm, float azmm);
  float getXmm() const;
  float getYmm() const;
  float getZmm() const;
  float getRotRad() const;
  float getLowRad() const;
  float getHighRad() const;
private:
  void calculateGrad();
  float xmm;
  float ymm;
  float zmm;
  float rot;
  float low;
  float high;
};

struct Point {
  float xmm;
  float ymm;
  float zmm;
  float emm;
};

class Interpolation {
public:
  //void resetInterpolation(float px, float py, float pz);
  //void resetInterpolation(float p1x, float p1y, float p1z, float p2x, float p2y, float p2z);
  //void resetInterpolation(Point p0, Point p1);
  
  void setCurrentPos(float px, float py, float pz, float pe);
  void setInterpolation(float px, float py, float pz, float pe, float v = 0);
  void setInterpolation(float p1x, float p1y, float p1z, float p1e, float p2x, float p2y, float p2z, float p2e, float av = 0);
  
  void setCurrentPos(Point p);
  void setInterpolation(Point p1, float v = 0);
  void setInterpolation(Point p0, Point p1, float v = 0);
  
  void updateActualPosition();
  bool isFinished() const;
  
  float getXPosmm() const;
  float getYPosmm() const;
  float getZPosmm() const;
  float getEPosmm() const;
  Point getPosmm() const;
  
private:
  float getDistance() const;
  byte state;
  
  long startTime;  
  
  float xStartmm;
  float yStartmm;
  float zStartmm;
  float eStartmm;
  float xDelta;
  float yDelta;
  float zDelta;
  float eDelta;
  float xPosmm;
  float yPosmm;
  float zPosmm;
  float ePosmm;
  float v;
  float tmul;
};

class FanControl {
public:
  FanControl(int aPin);
  void enable(bool value = true);
  void disable();
  void setDisableDelay(long millisec);
  void update();
private:
  bool state;
  int pin;
  long disableDelay;
  long nextShutdown;
};

class RampsStepper {
public:
  RampsStepper(int aStepPin, int aDirPin, int aEnablePin);
  void enable(bool value = true);
  void disable();
  
  bool isOnPosition() const;
  int getPosition() const;
  void setPosition(int value);
  void stepToPosition(int value);
  void stepRelative(int value);
  
  float getPositionRad() const;
  void setPositionRad(float rad);
  void stepToPositionRad(float rad);
  void stepRelativeRad(float rad);
  
  void update();
  
  void setReductionRatio(float gearRatio, int stepsPerRev);
private:
  int stepperStepTargetPosition;
  int stepperStepPosition;
    
  int stepPin;
  int dirPin;
  int enablePin;  
  
  float radToStepFactor;
};

template <typename Element> class Queue {
public:
  Queue(int alen);
  ~Queue();
  bool push(Element elem);
  Element pop();
  bool isFull() const;
  bool isEmpty() const;
  int getFreeSpace() const;
  int getMaxLength() const;
  inline int getUsedSpace() const;
private:
  Queue(Queue<Element>& q);  //copy const.
  Element* data;
  int len;
  int start;
  int count;
};


template <typename Element>
Queue<Element>::Queue(int alen) {
  data = new Element[alen];
  len = alen;
  start = 0;
  count = 0;
}

template <typename Element>
Queue<Element>::~Queue() {
  delete data;
}

template <typename Element>
Queue<Element>::Queue(Queue<Element>& q) {
  //nothing ever is allowed to do something here
}

template <typename Element>
bool Queue<Element>::push(Element elem) {
  data[(start + count++) % len] = elem;
  return true;
}

template <typename Element>
Element Queue<Element>::pop() {
  count--;
  int s = start;
  start = (start + 1) % len;
  return data[(s) % len];
}

template <typename Element>
bool Queue<Element>::isFull() const {
  return count >= len;
}

template <typename Element>
bool Queue<Element>::isEmpty() const {
  return count <= 0;
}

template <typename Element>
int Queue<Element>::getFreeSpace() const {
  return len - count;
}

template <typename Element>
int Queue<Element>::getMaxLength() const {
  return len;
}

template <typename Element>
int Queue<Element>::getUsedSpace() const {
  return count;
}

struct Cmd {
  char id;
  int num;
  float valueX;
  float valueY;
  float valueZ;
  float valueF;
  float valueE;
  float valueT; 
};

class Command {
  public:
    Command();
    bool handleGcode();
    bool processMessage(String& msg);
    Cmd getCmd() const;
  private: 
    int pos(String& s, char c, int start = 0);
    String message;
    Cmd command;
};

void printErr();
void printFault();
void printComment(char* c);
void printComment(String& s);
void printOk();

Stepper stepper(2400, STEPPER_GRIPPER_PIN_0, STEPPER_GRIPPER_PIN_1, STEPPER_GRIPPER_PIN_2, STEPPER_GRIPPER_PIN_3);
RampsStepper stepperRotate(Z_STEP_PIN, Z_DIR_PIN, Z_ENABLE_PIN);
RampsStepper stepperLower(Y_STEP_PIN, Y_DIR_PIN, Y_ENABLE_PIN);
RampsStepper stepperHigher(X_STEP_PIN, X_DIR_PIN, X_ENABLE_PIN);
RampsStepper stepperExtruder(E_STEP_PIN, E_DIR_PIN, E_ENABLE_PIN);
FanControl fan(FAN_PIN);
RobotGeometry geometry;
Interpolation interpolator;
Queue<Cmd> queue(15);
Command command;


void setup() {
  Serial.begin(9600);
  
  //various pins..
  pinMode(HEATER_0_PIN  , OUTPUT);
  pinMode(HEATER_1_PIN  , OUTPUT);
  pinMode(LED_PIN       , OUTPUT);
  
  //unused Stepper..
  pinMode(E_STEP_PIN   , OUTPUT);
  pinMode(E_DIR_PIN    , OUTPUT);
  pinMode(E_ENABLE_PIN , OUTPUT);
  
  //unused Stepper..
  pinMode(Q_STEP_PIN   , OUTPUT);
  pinMode(Q_DIR_PIN    , OUTPUT);
  pinMode(Q_ENABLE_PIN , OUTPUT);
  
  //GripperPins
  pinMode(STEPPER_GRIPPER_PIN_0, OUTPUT);
  pinMode(STEPPER_GRIPPER_PIN_1, OUTPUT);
  pinMode(STEPPER_GRIPPER_PIN_2, OUTPUT);
  pinMode(STEPPER_GRIPPER_PIN_3, OUTPUT);
  digitalWrite(STEPPER_GRIPPER_PIN_0, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_1, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_2, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_3, LOW);

  
  //reduction of steppers..
  stepperHigher.setReductionRatio(32.0 / 9.0, 200 * 16);  //big gear: 32, small gear: 9, steps per rev: 200, microsteps: 16
  stepperLower.setReductionRatio( 32.0 / 9.0, 200 * 16);
  stepperRotate.setReductionRatio(32.0 / 9.0, 200 * 16);
  stepperExtruder.setReductionRatio(32.0 / 9.0, 200 * 16);
  
  //start positions..
  stepperHigher.setPositionRad(PI / 2.0);  //90°
  stepperLower.setPositionRad(0);          // 0°
  stepperRotate.setPositionRad(0);         // 0°
  stepperExtruder.setPositionRad(0);
  
  //enable and init..
  setStepperEnable(false);
  interpolator.setInterpolation(0,120,120,0, 0,120,120,0);
  
  Serial.println("start");
}

void setStepperEnable(bool enable) {
  stepperRotate.enable(enable);
  stepperLower.enable(enable);
  stepperHigher.enable(enable); 
  stepperExtruder.enable(enable); 
  fan.enable(enable);
}

void loop () {
  //update and Calculate all Positions, Geometry and Drive all Motors...
  interpolator.updateActualPosition();
  geometry.set(interpolator.getXPosmm(), interpolator.getYPosmm(), interpolator.getZPosmm());
  stepperRotate.stepToPositionRad(geometry.getRotRad());
  stepperLower.stepToPositionRad (geometry.getLowRad());
  stepperHigher.stepToPositionRad(geometry.getHighRad());
  stepperExtruder.stepToPositionRad(interpolator.getEPosmm());
  stepperRotate.update();
  stepperLower.update();
  stepperHigher.update(); 
  fan.update();
  
  if (!queue.isFull()) {
    if (command.handleGcode()) {
      queue.push(command.getCmd());
      printOk();
    }
  }
  if ((!queue.isEmpty()) && interpolator.isFinished()) {
    executeCommand(queue.pop());
  }
    
  if (millis() %500 <250) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
}




void cmdMove(Cmd (&cmd)) {
  interpolator.setInterpolation(cmd.valueX, cmd.valueY, cmd.valueZ, cmd.valueE, cmd.valueF);
}
void cmdDwell(Cmd (&cmd)) {
  delay(int(cmd.valueT * 1000));
}
void cmdGripperOn(Cmd (&cmd)) {
  stepper.setSpeed(5);
  stepper.step(int(cmd.valueT));
  delay(50);
  digitalWrite(STEPPER_GRIPPER_PIN_0, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_1, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_2, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_3, LOW);
  //printComment("// NOT IMPLEMENTED");
  //printFault();
}
void cmdGripperOff(Cmd (&cmd)) {
  stepper.setSpeed(5);
  stepper.step(-int(cmd.valueT));
  delay(50);
  digitalWrite(STEPPER_GRIPPER_PIN_0, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_1, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_2, LOW);
  digitalWrite(STEPPER_GRIPPER_PIN_3, LOW);
  //printComment("// NOT IMPLEMENTED");
  //printFault();
}
void cmdStepperOn() {
  setStepperEnable(true);
}
void cmdStepperOff() {
  setStepperEnable(false);
}
void cmdFanOn() {
  fan.enable(true);
}
void cmdFanOff() {
  fan.enable(false);
}

void handleAsErr(Cmd (&cmd)) {
  printComment("Unknown Cmd " + String(cmd.id) + String(cmd.num) + " (queued)"); 
  printFault();
}

void executeCommand(Cmd cmd) {
  if (cmd.id == -1) {
    String msg = "parsing Error";
    printComment(msg);
    handleAsErr(cmd);
    return;
  }
  
  if (cmd.valueX == NAN) {
    cmd.valueX = interpolator.getXPosmm();
  }
  if (cmd.valueY == NAN) {
    cmd.valueY = interpolator.getYPosmm();
  }
  if (cmd.valueZ == NAN) {
    cmd.valueZ = interpolator.getZPosmm();
  }
  if (cmd.valueE == NAN) {
    cmd.valueE = interpolator.getEPosmm();
  }
  
   //decide what to do
  if (cmd.id == 'G') {
    switch (cmd.num) {
      case 0: cmdMove(cmd); break;
      case 1: cmdMove(cmd); break;
      case 4: cmdDwell(cmd); break;
      //case 21: break; //set to mm
      //case 90: cmdToAbsolute(); break;
      //case 91: cmdToRelative(); break;
      //case 92: cmdSetPosition(cmd); break;
      default: handleAsErr(cmd); 
    }
  } else if (cmd.id == 'M') {
    switch (cmd.num) {
      //case 0: cmdEmergencyStop(); break;
      case 3: cmdGripperOn(cmd); break;
      case 5: cmdGripperOff(cmd); break;
      case 17: cmdStepperOn(); break;
      case 18: cmdStepperOff(); break;
      case 106: cmdFanOn(); break;
      case 107: cmdFanOff(); break;
      default: handleAsErr(cmd); 
    }
  } else {
    handleAsErr(cmd); 
  }
}