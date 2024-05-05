from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import json
import time

app = Flask(__name__)

# Initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("D:\\data analytics\\REAL-TIME PROJECTS\\actonlexa\\new\\model\\keras_model.h5", "D:\\data analytics\\REAL-TIME PROJECTS\\actonlexa\\new\\model\\labels.txt")
offset = 20
imgSize = 300
labels = ["Hi Hello", "How are you", "Nice to meet you after long time", "Superb", "Where do you live right now", "Ohh okay", "See you soon"]

# List to store index and label values
gesture_data = []
label_history = []  # List to store labels corresponding to index

def gen_frames():
    global gesture_data, label_history
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # Append index value and label to the list
                gesture_data.append((index, labels[index]))
                label_history.append(labels[index])  # Append the label to the history list
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        # Printing label history on console
        # print("Label History:", label_history)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_label_updates():
    global label_history
    while True:
        if label_history:
            # Send the latest label in SSE format
            yield f"data: {label_history[-1]}\n\n"
        time.sleep(1)  # Adjust the sleep time as needed

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/aboutUs')
def about():
    return render_template('aboutUs.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/label_updates')
def label_updates():
    return Response(generate_label_updates(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)
