from ast import arg
from lib2to3.pgen2.driver import Driver

import threading
from turtle import distance

from Violation_Detection.safety_sys import SafetySystem
from Alarm.alarm import Alarm
from init_vars import firebase_interface, driver_id
from DriverRater import RatingModule
from alertcodeyolo import ObjectDetection
import cv2
from time import time





class Car():
    def __init__(self,speed=None, distance=None) -> None:
        self.speed = speed
        self.distance = distance
        
        
    def set_ID(self,ID):
        self.ID = ID
    def set_speed(self,speed):
        self.speed = int(speed)
    def set_distance(self,distance):
        self.distance = int(distance)
    def print_car_status(self):
        print(f"Speed: {self.speed} \nDistance : {self.distance}")

        


            
class CarSystem(Car):

    def __init__(self,speed=None, distance=None):
        Car.__init__(self,speed,distance)
        self.safety_system = SafetySystem()
        self.drowsiness_detector = ObjectDetection(0)
        threading.Thread(target=self.init_drowsiness_detector, daemon=True).start()
        self.DROWSINESS_GRACE_PERIOD = 3
        self.prev_drowsiness_violation_time = 0
        self.alert_triggered = False
        self.rating_module = RatingModule(firebase_interface)

    
    def init_drowsiness_detector(self):
        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:

            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.drowsiness_detector.predict(im0)
            self.drowsiness_detector.detectAction(results)
            im0, class_ids = self.drowsiness_detector.plot_bboxes(results, im0)
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            for cls in clss:
                if names[int(cls)] == "drowsy":
                    self.drowsiness_detector.counter += 1
                    if (self.drowsiness_detector.counter > self.drowsiness_detector.DROWSY_THRESHOLD) :
                        if not self.alert_triggered:  # Prevent multiple alerts per frame
                            print("ALERT")
                            self.alert_triggered = True
                            # beep_thread = threading.Thread(target=self.drowsiness_detector.alarm.fire_sound_alarm)
                            # beep_thread.daemon = True  # This allows the thread to exit when the main program exits
                            # beep_thread.start()
                            

                            if ((self.start_time - self.prev_drowsiness_violation_time > self.DROWSINESS_GRACE_PERIOD)):
                                # Log Violation
                                self.safety_system.log_drowsiness_violation()

                                # Update Rating
                                self.rating_module.update_driver_rating(driver_id)

                                self.prev_drowsiness_violation_time = self.start_time
                else:
                    self.counter = 0
            self.alert_triggered = False
            #self.drowsiness_detector.display_fps(im0)
            #self.drowsiness_detector.update_drowsiness_th(self.drowsiness_detector.fps)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        

        
    def open_server_for_car(self):
        ...
    def update_speed(self,speed):
        self.speed = speed
        is_speed_violation = self.safety_system.detect_speed_violation(self.speed)
        if(is_speed_violation):
            # Update rating
            self.rating_module.update_driver_rating(driver_id)
        firebase_interface.update_vehicle_speed(driver_id,speed);
    def update_distance(self,distance):
        self.distance = distance

        #Detect Distance Violation
        is_distance_violation = self.safety_system.detect_distance_violation(self.distance)

        # update measurements
        firebase_interface.update_vehicle_distance(driver_id,distance);
        if(is_distance_violation):
            # Update rating
            self.rating_module.update_driver_rating(driver_id)

