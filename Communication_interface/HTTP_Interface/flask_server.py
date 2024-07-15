import sys
import init
import threading
import requests, json
from flask import Flask, request, jsonify
from car import Car, CarSystem
from flasgger import Swagger

car_sys = CarSystem()

class VehicleInterface():
    app = Flask(__name__)
    swagger = Swagger(app)

    def __init__(self, port=80) -> None:
        self.port = port

    @app.route('/set_speed', methods=['GET'])
    def set_speed():
        """
        Set the speed of the car
        ---
        parameters:
          - name: speed
            in: query
            type: integer
            required: true
            description: The speed to set for the car
        responses:
          200:
            description: Speed set successfully
            schema:
              type: object
              properties:
                message:
                  type: string
                status:
                  type: string
          400:
            description: Error occurred
        """
        speed = request.args.get('speed')
        if speed is not None:
            print(f'speed = {speed}')
            speed = int(speed)
            car_sys.update_speed(speed)
            return jsonify({"message": "Speed set successfully", "status": "success"})
        else:
            return jsonify({"message": "some error occurred", "status": "fail"})

    @app.route('/set_distance', methods=['GET'])
    def set_distance():
        """
        Set the distance from other cars
        ---
        parameters:
          - name: distance
            in: query
            type: integer
            required: true
            description: The distance to set from other cars
        responses:
          200:
            description: Distance set successfully
            schema:
              type: object
              properties:
                message:
                  type: string
                status:
                  type: string
          400:
            description: Error occurred
        """
        distance = request.args.get('distance')
        if distance is not None:
            print(f"request: {request}")
            print(f"distance from other cars: {distance}")
            distance = int(distance)
            car_sys.update_distance(distance)
            return jsonify({"message": "distance set successfully", "status": "success"})
        else:
            return jsonify({"message": "some error occurred", "status": "fail"})

    @app.route('/', methods=['GET', 'POST'])
    def get_status():
        """
        Used for testing connection
        ---
        parameters:
          - name: test
            in: query
            type: string
            required: false
            description: Test data
        responses:
          200:
            description: Status information
            schema:
              type: object
              properties:
                message:
                  type: string
        """
        test_data = request.args.get('test')
        print(test_data)
        print("Passed on default route")
        return jsonify({"message": "GET not implemented yet"})

    def open_connection(self):
        threading.Thread(target=self.app.run, daemon=True, args=(None, self.port)).start()
        # self.app.run(host="0.0.0.0", port=2156)

if __name__ == '__main__':
    car = CarSystem()
    car_interface = VehicleInterface(car, 2156)
    car_interface.open_connection()
    # threading.Thread(target=app.run, daemon=True).start()
    while(True):
        if input() == ord('Q'):
            sys.exit(0)


    


