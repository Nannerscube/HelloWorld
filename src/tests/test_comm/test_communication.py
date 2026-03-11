#import arduino.app_utils as app # should be the same as below
from arduino import bridge # should be the same as above
import time

def setup():
    print("setup complete")
    time.sleep(2)  # wait for bridge init

def main():
    setup()
    
    app.bridge.call("fromC")  # call c function

if __name__ == "__main__":
    main()
