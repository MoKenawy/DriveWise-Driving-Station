import sys
import os 
import init
from DriverRater import RatingModule
from Communication_interface.Firebase_interface.firebase_interface import FirebaseInterface
import os

from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv('DB_URL')
creds = os.getenv('SERVICE_ACCOUNT_FILE')
drowsiness_model = os.getenv('YOLO_DROWSINESS_DETECTION_MODEL')

# # Initialize Firebase interface
# db_url = "https://graduation-project-3fd49-default-rtdb.firebaseio.com/"
# # creds = "Communication interface\\Firebase_interface\\Creds\\graduation-project-Mostafa-Service_key.json"
# creds = "G:\\SJ\\Project\\Self-Driving Car\\Safe Driving Assistant\\Driving Station\\Communication interface\\Firebase_interface\\Creds\\graduation-project-Mostafa-Service_key.json"

firebase_interface = FirebaseInterface(creds, db_url)
driver_id = 'Ali'

# drowsiness_model = 'G:\\SJ\\Project\\Self-Driving Car\\Safe Driving Assistant\\Driving Station\\Violation_Detection\\Drowsiness_Detection\\Yolo\\best.pt'
