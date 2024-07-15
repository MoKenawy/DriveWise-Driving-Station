import sys
import init
from Communication_interface.HTTP_Interface.flask_server import VehicleInterface

if __name__ == '__main__':
    car_interface = VehicleInterface(port = 2156)
    car_interface.open_connection()
    while(True):
        if input() == ord('Q'):
            sys.exit(0)