#!/usr/bin/python

# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
from mfrc522 import SimpleMFRC522
import RPi.GPIO as GPIO
import threading

RELAY = 21
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY, GPIO.HIGH)

# RFID Reader initialization
reader = SimpleMFRC522()

# Define the valid UIDs as long integers
valid_uids = [
    566100745643,
    357683295863,  # UID of Nurse1
    # Add more UIDs here
]

nurse_names = [
    "Valhari Meshram",  # Nurse 1
    "Vishakha Fulare",  # Nurse 2
    # Add more names here
]

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# Use this xml file
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade = "haarcascade_frontalface_default.xml"

# Load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS counter
fps = FPS().start()

prevTime = 0
doorUnlock = False

# Initialize a threading Event
face_recognition_event = threading.Event()

# Function to handle RFID scanning
def rfid_scan():
    global doorUnlock, prevTime, currentname
    while True:
        # Wait for the event to be set, indicating a face has been recognized
        face_recognition_event.wait()
        face_recognition_event.clear()
        print("Ready to scan RFID tag")
        uid, text = reader.read()
        if uid in valid_uids:
            print("Access Granted")
            print("Welcome Nurse", nurse_names[valid_uids.index(uid)])
            GPIO.output(RELAY, GPIO.LOW)  # Turn on the relay
            print("door unlock")
            time.sleep(10)
            GPIO.output(RELAY, GPIO.HIGH) 
            print("door lock")
            
        else:
            print("Access Denied")
            GPIO.output(RELAY, GPIO.HIGH)  # Turn off the relay
            time.sleep(2)

# Create a thread for RFID scanning
rfid_thread = threading.Thread(target=rfid_scan)
rfid_thread.daemon = True  # Set the thread as a daemon to exit when the main program exits
rfid_thread.start()

# Loop over frames from the video file stream
while True:
    # Grab the frame from the threaded video stream and resize it
    # to 500px (to speed up processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # Compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop over the facial embeddings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"  # If the face is not recognized, then print Unknown

        # Check to see if we have found a match
        if True in matches:
            # Find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # Loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie, Python
            # will select the first entry in the dictionary)

            name = max(counts, key=counts.get)

            # If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                currentname = name
                print(currentname)
                # Set the event to signal that a face has been recognized
                face_recognition_event.set()

        # Update the list of names
        names.append(name)
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF 

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do some cleanup
cv2.destroyAllWindows()
vs.stop()