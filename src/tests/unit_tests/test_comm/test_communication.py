from arduino.app_utils import App, Bridge
import time

def setup():
    print("setup complete")
    time.sleep(2)  # wait for bridge init

def main():
    setup()
    app = App()
    app.bridge.call("fromC")  # call c function

if __name__ == "__main__":
    main()
