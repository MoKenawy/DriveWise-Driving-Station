from ast import main
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import joblib
import warnings
warnings.filterwarnings("ignore")
import sys
import asyncio
import sounddevice as sd
import numpy as np
import threading
import time
import tensorflow as tf
from numba import cuda
#from gamma_correction import adjust_gamma
import os

sys.path.append("g:\\SJ\\Project\\Self-Driving Car\\Safe Driving Assistant\\Driving Station")
import init
# from feature_extraction.calc_features import eye_aspect_ratio, mouth_aspect_ratio, compute_perclos,  eye_circularity, level_of_eyebrows, size_of_pupil
# from feature_extraction.extract_features import construct_eyes, construct_mouth, construct_eyebrows 
from feature_extraction.facial_landmarks_dlib import FacialLandmarks
from feature_extraction.facial_features_dlib import FacialFeatures
from feature_extraction.extract_features import display_features_onframe
from Alarm.alarm import Alarm

class DrowsinessDetector():
    def __init__(self,EAR_THRESHOLD=0.21, CONSEC_THRESHOLD=3):
        self.EAR_THRESHOLD = EAR_THRESHOLD
        self.CONSEC_THRESHOLD = CONSEC_THRESHOLD    
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("Violation_Detection/Drowsiness_Detection/models/shape_predictor_68_face_landmarks.dat")
        self.loaded_model = tf.keras.models.load_model('Violation_Detection\\Drowsiness_Detection\\models\\saved_model\\my_model')
        self.alarm = Alarm()
        self.feature_extractor = FacialFeatures()


    def predict_webcam(self, record=False, record_path=None):
        
        COUNTER = 0
        SELECTED_FRAMES = 15
        vid = cv2.VideoCapture(0)
        shapes = []  
        # Caclulate time
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        fps_calc = -1
        print(f"FPS of the video: {fps}")
        start_time = time.time()
        frame_count = 0
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the output video
        out = cv2.VideoWriter(f'{record_path}.avi', fourcc, 10, (240, 180))  # Output file name, codec, frame rate, and frame size

        while(True):
            ret, frame = vid.read() 
            if frame is not None:

                frame_count += 1
                height = frame.shape[0]
                width = frame.shape[1]
                print(int(240 * height / width))
                frame = cv2.resize(frame, (240, 180))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                
                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)

                for (i, rect) in enumerate(rects):          
                    shape = self.predictor(gray, rect)
                    shape = np.array([[p.x, p.y] for p in shape.parts()])
                    if len(shapes) < SELECTED_FRAMES:
                        shapes.append(shape)
                    elif len(shapes) >= SELECTED_FRAMES:           
                        ear,mar, moe, eye_circ,leb, sop, closeness, blink_no, perclos = self.calc_feature_vector(shapes)
                        state = self.predict(ear,mar, moe, eye_circ,leb, sop, closeness, blink_no, perclos)
                        if state == 1:
                            cv2.putText(frame, "ALERT: WAKE UP", (int(frame.shape[1]/2), int(frame.shape[0]/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            beep_thread = threading.Thread(target=self.alarm.fire_sound_alarm)
                            beep_thread.daemon = True  # This allows the thread to exit when the main program exits
                            beep_thread.start()
                        
                    # # convert dlib's rectangle to a OpenCV-style bounding box
                    # # [i.e., (x, y, w, h)], then draw the face bounding box
                    #     (x, y, w, h) = face_utils.rect_to_bb(rect)
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # # show the face number
                    #     cv2.putText(frame, f"State #{state}", (x - 10, y - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw them on the image
                        for (x,y) in shape:
                            cv2.circle(frame, (x,y), 1, (0, 0, 255), -1)
                        shapes.pop(0)
                    # Display features
                    #display_features_onframe(frame, ear, eye_circ, mar,leb, sop, perclos)

                
                if COUNTER %30 ==0:
                    print("pass")
                    COUNTER = 0
                COUNTER += 1
                elapsed_time = time.time() - start_time
                # Calculate FPS every second
                if elapsed_time > 1:
                    fps_calc = frame_count / elapsed_time
                    print(f"Calculated FPS: {fps_calc:.2f}")
                    # Reset variables for the next second
                    elapsed_time = 0
                    start_time = time.time()
                    frame_count = 0
                # Display FPS on the frame
                cv2.putText(frame, f"FPS: {int(fps_calc)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('frame', frame) 
                #Save video
                if record == True:
                    out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        out.release()  
        vid.release() 
        cv2.destroyAllWindows() 

    def predict(self,ear,mar, moe, eye_circ,leb, sop, closeness, blink_no, perclos):
        feature_names =['mar', 'eye_circularity',
        'leb', 'sop', 'closeness' , 'perclos']
        feature_vector = [[ear,mar, moe, eye_circ,leb, sop, closeness, blink_no, perclos]]
        state_prob = self.loaded_model.predict(feature_vector, verbose=0)
        state = (state_prob > 0.5).astype(int) 
        return state

    def display_features_onframe(self,frame, ear , eye_circ, mar,  leb, sop, perclos):
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "PERCLOS: {:.2f}".format(perclos), (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "MOE: {:.2f}".format(moe), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EC: {:.2f}".format(eye_circ), (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LEB: {:.2f}".format(leb), (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "SOP: {:.2f}".format(sop), (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    def calc_ear_test(self, shapes):
        for shape in shapes:
            facial_areas = FacialLandmarks(shape)
            leftEye, rightEye = facial_areas.construct_eyes()
            leftEAR = FacialFeatures.eye_aspect_ratio(leftEye)
            rightEAR = FacialFeatures.eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            return ear

    def calc_feature_vector(self, shapes):
        perclos_list = []
        COUNTER = 0
        blink_no = 0
        for shape in shapes:
            facial_areas = FacialLandmarks(shape)
            leftEye, rightEye = facial_areas.construct_eyes()
            leftEAR = FacialFeatures.eye_aspect_ratio(leftEye)
            rightEAR = FacialFeatures.eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            mouth = facial_areas.construct_mouth()
            mar = FacialFeatures.mouth_aspect_ratio(mouth)
            # #mouth over eye
            moe = mar/ear

            #eye circularity
            leftEC = FacialFeatures.eye_circularity(leftEye)
            rightEC = FacialFeatures.eye_circularity(rightEye)
            eye_circ = (leftEC + rightEC) / 2.0

            left_eye_leb_coordinates, right_eye_leb_coordinates = facial_areas.construct_eyebrows()
            leftEyeLEB = FacialFeatures.level_of_eyebrows(left_eye_leb_coordinates)
            rightEyeLEB = FacialFeatures.level_of_eyebrows(right_eye_leb_coordinates)
            leb = (rightEyeLEB + leftEyeLEB) / 2

            #size of pupil
            leftEyeSOP = FacialFeatures.size_of_pupil(leftEye)
            rightEyeSOP = FacialFeatures.size_of_pupil(rightEye)
            sop = (leftEyeSOP + rightEyeSOP) / 2
            # capturing blink
            if ear < self.EAR_THRESHOLD:
                perclos_list.append(1)
                # COUNTER += 1
                closeness = 1
            else:
                perclos_list.append(0)
                if COUNTER >= self.CONSEC_THRESHOLD:
                    blink_no += 1
                #COUNTER = 0
                closeness = 0
        perclos = self.feature_extractor.compute_perclos(perclos_list)
        return ear,mar, moe, eye_circ,leb, sop, closeness, blink_no, perclos



    def test_yawdd(self,images_dir):  
        SELECTED_FRAMES = 15
        shapes = []
        history = []
        files = os.listdir(images_dir)
        for file in files[:10]:
            image_path = os.path.join(images_dir, file)
            frame = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            if frame is not None:

                height = frame.shape[0]
                width = frame.shape[1]
                frame = cv2.resize(frame, (240, 180))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                
                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)
                for (i, rect) in enumerate(rects):          
                    shape = self.predictor(gray, rect)
                    shape = np.array([[p.x, p.y] for p in shape.parts()])
                    ear = self.calc_ear_test(shapes)
                    if ear != None and ear <= self.EAR_THRESHOLD:
                        state = 1
                    else:
                        state = 0
                    history.append(state)
                    print(state)
                    if ear != None:
                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('image', frame)
                    cv2.waitKey()
        cv2.destroyAllWindows()
        true_pos = [x for x in history if x==1]
        print(len(true_pos)/ len(history))
        return history


if __name__ == "__main__":
    print(cuda.gpus)
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU")
        # Force TensorFlow to use GPU
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    else:
        print("GPU not available")
    drowsiness_detector = DrowsinessDetector()
    # drowsiness_detector.predict_webcam(record=True,record_path="drowsiness_dlib_30_FPS")
    path = "M:\\DDD-Datasets\\dataset_B_FacialImages_highResolution\\dataset_B_FacialImages_highResolution"
    drowsiness_detector.test_yawdd(path)

