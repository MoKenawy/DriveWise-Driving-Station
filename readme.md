# Drive-Wise (Driving station)


## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)

## Introduction

Drive-Wise (Driving station) is a comprehensive system designed to enhance vehicle monitoring and driver safety. The project facilitates communication with a vehicle through a REST API to retrieve real-time speed and distance data, and provides remote control capabilities. It includes advanced detection of speed, distance, and drowsiness violations, and alerts the driver with an auditory alarm. Violations are logged on Firebase, and the driver's rating is automatically updated based on their performance. This system aims to improve driving habits and ensure road safety through continuous monitoring and instant feedback.

## Features

- Communicate with a vehicle through a REST API to get the speed and distance of the vehicle.
- Control the vehicle remotely.
- Detect speed, distance, and drowsiness violations.
- Alarm the driver using an auditory alarm.
- Log driver violations on Firebase and update the rating automatically.

## Dependencies
- Python 3.10.8 64-bit

### API
- Flask
- Flassger

### Database
- firebase_admin

### Drowsiness Detection
- OpenCV
- Ultralytics
- torch
- torchvision
- Numpy

### Alarm
- sounddevice

### Profiling
- memory_profiler
- GPUtil
- psutil

## Usage
"init_vars" file at the root directory is used to initialze FirebaseInterface instance and the driver_id for testing.

The main entry to the program is through the file named `Communication_Interface/Http_Interface/flask_server`. The example below demonstrates how to start the program. Note that `2156` is an example port number; you can choose any free port not reserved on your machine.

```python
if __name__ == '__main__':
    car = CarSystem()
    car_interface = VehicleInterface(car, 2156)  # Replace 2156 with any free port number
    car_interface.open_connection()
    while(True):
        if input() == ord('Q'):
            sys.exit(0)
```
go to "http://localhost/2156/apidocs" to try the API using Swagger UI


