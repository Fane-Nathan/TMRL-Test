
import time
import struct
import numpy as np
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

def inspect_data():
    client = TM2020OpenPlanetClient()
    print("Listening for data... Drive in the game!")
    print("Columns: [Speed, ?, ?, ?, ?, ?, ?, ?, ?, Gear, RPM, ?]")
    
    try:
        while True:
            data = client.retrieve_data(sleep_if_empty=0.01)
            # data is a tuple of 11 floats
            # Let's print them formatted
            formatted = [f"{x:.2f}" for x in data]
            print(f"Data: {formatted}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping inspection.")

if __name__ == "__main__":
    inspect_data()
