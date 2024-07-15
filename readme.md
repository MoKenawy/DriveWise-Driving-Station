# Drive-Wise (Driving station)


## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [License](#license)
7. [Related Repositories](#related-repositories)
8. [Contact](#contact)

## Introduction

Drive-Wise (Driving station) is a comprehensive system designed to enhance vehicle monitoring and driver safety. The project facilitates communication with a vehicle through a REST API to retrieve real-time speed and distance data, and provides remote control capabilities. It includes advanced detection of speed, distance, and drowsiness violations, and alerts the driver with an auditory alarm. Violations are logged on Firebase, and the driver's rating is automatically updated based on their performance. This system aims to improve driving habits and ensure road safety through continuous monitoring and instant feedback.

## Features

- Communicate with a vehicle through a REST API to get the speed and distance of the vehicle.
- Control the vehicle remotely.
- Detect speed, distance, and drowsiness violations.
- Alarm the driver using an auditory alarm.
- Log driver violations on Firebase and update the rating automatically.

## Dependencies
- Python 3.12.1 64-bit

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



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MoKenawy/DriveWise-Driving-Station.git
   cd DriveWise-Driving-Station
   ```

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   cd DriveWiseProject
   py main.py
   ```
## Configuration

Create a `.env` file in the root directory and add the following configurations:

```env
DB_URL=your_database_url
SERVICE_ACCOUNT_FILE=path_to_your_service_account_file
YOLO_DROWSINESS_DETECTION_MODEL=path_to_your_yolo_drowsiness_detection_model
```

- **DB_URL**: The URL of your firebase real-time database.
- **SERVICE_ACCOUNT_FILE**: The path to your Firebase service account file.
- **YOLO_DROWSINESS_DETECTION_MODEL**: The path to your YOLO model for drowsiness detection.

## Usage
"init_vars" file at the root directory is used to initialze FirebaseInterface instance and the driver_id for testing.

The main entry to the program is through the file named `main.py`. The example below demonstrates how to start the program. Note that `2156` is an example port number; you can choose any free port not reserved on your machine.

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


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Related Repositories

- [Manager's Dashboard](https://github.com/MoKenawy/Drive-Wise-Django): Provides tools for fleet managers to monitor and analyze driver performance.

## Contact

For any questions or feedback, please contact:
- Mohammed Gamal: [mokenawy.business@gmail.com](mailto:mokenawy.business@gmail.com)
