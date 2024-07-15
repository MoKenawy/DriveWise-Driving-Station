import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import sys
import os
import init

class FirebaseInterface:
    def __init__(self, credentials_path, db_url):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {'databaseURL': db_url})
        self.ref = db.reference()

    def log_violation(self, driver_id, violation_type, violation_details):
        timestamp = datetime.now().isoformat()
        violation_data = {
            'time': timestamp,
            'type': violation_type,
            'details': violation_details
        }
        self.ref.child('drivers').child(driver_id).child('violations').push(violation_data) # type: ignore
    def update_vehicle_status(self, driver_id, current_speed, current_distance):
        vehicle_data = {
            'current_speed': current_speed,
            'current_distance': current_distance
        }
        self.ref.child('drivers').child(driver_id).child('vehicle').update(vehicle_data)
    def update_vehicle_speed(self, driver_id, current_speed):
        vehicle_data = {
            'current_speed': current_speed        }
        self.ref.child('drivers').child(driver_id).child('vehicle').update(vehicle_data)
    def update_vehicle_distance(self, driver_id, current_distance):
        vehicle_data = {
            'current_distance': current_distance
        }
        self.ref.child('drivers').child(driver_id).child('vehicle').update(vehicle_data)



    # Add other methods as needed
    
    def update_driver_rating(self, driver_id, rating):
        self.ref.child('drivers').child(driver_id).update({'rating': rating})

    def get_driver_data(self, driver_id):
        return self.ref.child('drivers').child(driver_id).get()



# class ViolationLogger:
#     def __init__(self, firebase_interface):
#         self.firebase_interface = firebase_interface

#     def log_violation(self, violation):
#         self.firebase_interface.log_violation(violation['driverID'], violation['type'])


class MeasurementsUpdater:
    def __init__(self, firebase_interface):
        self.firebase_interface = firebase_interface

    def update_measurements(self, car_id, speed, distance):
        self.firebase_interface.update_measurements(car_id, speed, distance)


