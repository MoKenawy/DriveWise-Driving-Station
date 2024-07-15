import threading
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import sys
sys.path.append("g:\\SJ\\Project\\Self-Driving Car\\Safe Driving Assistant\\Driving Station")
import init
from Alarm.alarm import Alarm
from torchvision import transforms
from init_vars import drowsiness_model


class ObjectDetection:
    fps = -1
    DROWSY_THRESHOLD = 15

    def __init__(self, capture_index):
        
        self.capture_index = capture_index
        #self.alert_condition = alert_condition 
        #self.alert_function = alert_function  # Function to trigger alert
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model = YOLO(drowsiness_model)

        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.counter = 0
        self.alarm = Alarm()
        self.alert_triggered = False


        
    def predict(self, im0):
        
        results = self.model(im0, show_conf=True, verbose = False)
        return results

    def display_fps(self, im0):
        self.end_time = time.time()
        if (self.end_time == self.start_time):
            self.fps = 0
            return 0
        self.fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(self.fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def update_drowsiness_th(self,fps, time_th = 0.5):
        self.DROWSY_THRESHOLD = int(np.round(self.fps * time_th))

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for cls in clss:
            if names[int(cls)] == "drowsy":
                self.counter += 1
                if (self.counter > self.DROWSY_THRESHOLD) :
                    if not self.alert_triggered:  # Prevent multiple alerts per frame
                        #self.alert_function("D:\\test\\emergency-alarm-with-reverb-29431 (1).mp3")  # put sound alert 
                        # print("ALERT : Drowsiness detected")
                        self.alert_triggered = True
                        beep_thread = threading.Thread(target=self.alarm.fire_sound_alarm)
                        beep_thread.daemon = True  # This allows the thread to exit when the main program exits
                        beep_thread.start()
            else:
                self.counter = 0
   
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            label =  "True Drowsiness" if self.alert_triggered == True else names[int(cls)]
            self.annotator.box_label(box, label, color=colors(int(cls), True))

        self.alert_triggered = False  # Reset flag for next frame

        return im0, class_ids

    def detectAction(self,results):
        drowsy_confidence_history = []
        states = results[0].boxes.cls
        for i in range(len(states)):
            current_state = states[i]
            if int(current_state) == 1:
                confidence = results[0].boxes.conf[i]
                drowsy_confidence_history.append(confidence)
                # print(f"state: {current_state}\nconfidence: {confidence}")
                # print(f"average drowsy confidence: {np.mean(drowsy_confidence_history)}")

    def image_to_tensor(self,img):
        # img = torch.from_numpy(img).to(self.device)  # Ensure your image tensor is also on GPU
        transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        img = transform(img).unsqueeze(0).to(self.device)
        return img
                
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time.time()
            ret, im0 = cap.read()
            # im0 = self.image_to_tensor(im0)
            assert ret
            results = self.predict(im0)
            self.detectAction(results)

            im0, class_ids = self.plot_bboxes(results, im0)


            self.display_fps(im0)
            self.update_drowsiness_th(self.fps)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        
                


def test_yawdd(images_dir):
    yolo = ObjectDetection(0)
    files = os.listdir(images_dir)
    for file in files[:10]:
        image_path = os.path.join(images_dir, file)
        frame = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame, (640, 480))
        results = yolo.predict(frame)
        frame, class_ids = yolo.plot_bboxes(results, frame)
        cv2.imshow('YOLOv8 Detection', frame)
        cv2.waitKey()
    cv2.destroyAllWindows()
        


if __name__ == "__main__":
    #print("Cuda Enabled : " + torch.cuda.is_available())
    detector = ObjectDetection(0)
    detector.__call__()
    # path = "M:\\DDD-Datasets\\dataset_B_FacialImages_highResolution\\dataset_B_FacialImages_highResolution"
    # test_yawdd(path)

