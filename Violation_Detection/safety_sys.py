import sys
import os 
import init
from time import time
from Alarm.alarm import Alarm
from init_vars import firebase_interface, driver_id

class SafetySystem():
    def __init__(self,speed_threshold = 60, distance_threshold = 3):
        self.speed_threshold = speed_threshold
        self.distance_threshold = distance_threshold
        self.alarm = Alarm()
        self.prev_speed_violation_time = 0
        self.prev_distance_violation_time = 0
        self.DEACCELERATE_PERIOD = 3
    
    def detect_speed_violation(self,speed):
        current_time = time()
        if (((current_time - self.prev_speed_violation_time) > self.DEACCELERATE_PERIOD)):
            if speed > self.speed_threshold:
                self.alarm.fire_alarm()
                self.alarm.fire_sound_alarm()
                self.log_speed_violation(speed, self.speed_threshold)
                self.prev_speed_violation_time = current_time
                return True
            else:
                return False
    def log_speed_violation(self,speed,threshold):        
        speed_violation_details = {
            'car_speed': speed,
            'speed_threshold': threshold
        }
        firebase_interface.log_violation(driver_id, 'speed_violation', speed_violation_details)
    


    def detect_distance_violation(self,distance):
        current_time = time()
        if (((current_time - self.prev_distance_violation_time) > self.DEACCELERATE_PERIOD)):            
            if distance < self.distance_threshold:
                self.alarm.fire_alarm()
                self.alarm.fire_sound_alarm()
                self.log_distance_violation(distance)
                self.prev_distance_violation_time = current_time

                return True
            else:
                return False
    def log_distance_violation(self,distance):        
        distance_violation_details = {
            'distance': distance,
        }
        firebase_interface.log_violation(driver_id, 'distance_violation', distance_violation_details)
    

    def detect_all_violations(self,speed,distance):
        is_speed_violation = self.detect_speed_violation(speed)
        is_distance_violation = self.detect_distance_violation(distance)
        return is_speed_violation or is_distance_violation
    



    def log_drowsiness_violation(self,image = "#TO-DO"):
        
        distance_violation_details = {
            'image': image,
        }
        firebase_interface.log_violation(driver_id, 'drowsiness_violation', distance_violation_details)
        ...
