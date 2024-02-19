#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import pyttsx3


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    engine = pyttsx3.init()

    while alarm_status:
        print('call')
        engine.say(msg)
        engine.runAndWait()

    if alarm_status2:
        print('call')
        saying = True
        engine.say(msg)
        engine.runAndWait()
        saying = False




def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
#detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('D:\\my_projects\\drowsiness\\code\\new\\shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

output_folder = 'output_frames'  # Change this to your desired folder
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

frame_count = 0  # Counter for the frame number
start_time = time.time()  # Start time for tracking


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    for rect in rects:
        # No need to redefine rect, it's already a dlib.rectangle object

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(eye[1])
        rightEyeHull = cv2.convexHull(eye[2])
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Rest of the code remains unchanged

    # cv2.imshow("Frame", frame)
    
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Capture 2 images per second
    if elapsed_time >= 0.25:  # 0.5 seconds for each image
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        start_time = time.time()  # Reset the start time

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
# out.release()
cv2.destroyAllWindows()
vs.stop()
