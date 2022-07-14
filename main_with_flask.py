import cv2
import numpy as np
import pandas as pd
import time
from flask import Flask, request, render_template, Response, send_file, session
from easyocr import Reader
import camera
from csv import writer
import os
import warnings
import tablib

app=Flask(__name__)

@app.route('/')
def home1():
    return render_template('html_main.html')

# Load Yolo
net = cv2.dnn.readNet("weights/yolov4.weights", "cfg/yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Loading camera
cap = cv2.VideoCapture('video_test.mp4')

#cap.get(cv2.CAP_PROP_FPS)
font = cv2.FONT_HERSHEY_PLAIN

starting_time = time.time()
reader = Reader(['en'])
max_plat_no = set()

@app.route('/start',methods=['GET','POST'])
def start():
    frame_id = 0
    while True:
        ret, frame = cap.read()

        frame_id += 1

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                # cv2.rectangle(frame, (x, y), (x + w, y + 30), color)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 10), font, 2, (255, 255, 255),2)
                # extract number plate area in another window
                crop_image = frame[y:y + h, x:x + w]
                #grayscale the image
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                # thresh the image using otus method
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
                #apply the easyocr to the thresh
                text = reader.readtext(gray)

                #print(ret)
                if len(text) > 0:
                    if (text[0][2]) > 0.80:
                        print(text[0][1])
                        if text[0][1] in max_plat_no:
                            pass
                        else:
                            max_plat_no.add(text[0][1])
                            with open('Number_Record.csv', 'a+') as csv_file:
                                csv_file.write(text[0][1] + '\n')
                    cv2.imshow("crop",crop_image)

        elapsed_time = time.time() - starting_time

        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps,1)), (10, 50), font, 3, (0, 0, 0), 2)
        cv2.imshow("Frame Rate", frame)

        #fram = jpeg.tobytes()

        #yield (b'--fram\r\n'
         #      b'Content-Type: frame/jpeg\r\n\r\n' + fram + b'\r\n\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('html_main.html')


@app.route('/stop',methods=['POST'])
def stop():
    cap.release()
    cv2.destroyAllWindows()
    return render_template('html_main.html')


#to run the flask app
if __name__=='__main__':
    app.run(debug=True,port=6664)

