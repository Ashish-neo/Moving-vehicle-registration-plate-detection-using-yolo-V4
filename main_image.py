import cv2
import numpy as np
import re
import easyocr
from easyocr import Reader
from flask import Response
from flask import Flask
from flask import render_template
#initialize a flask object
app=Flask(__name__)

#load the yolov4 model
Net=cv2.dnn.readNet("weights/yolov4.weights","cfg/yolov4.cfg")
classes=[]
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names=Net.getLayerNames()
output_layers = [layer_names[i-1] for i in Net.getUnconnectedOutLayers()]
colors=np.random.uniform(0,255,size=(len(classes),1))

#loading image
img=cv2.imread("car1.png")
img=cv2.resize(img,None,fx=0.4,fy=0.4)
height,width,channels=img.shape

#detecting number plate
blob=cv2.dnn.blobFromImage(img, 0.00392,(416,416),(0,0,0),False,crop=False)

Net.setInput(blob)
outs=Net.forward(output_layers)

#showing on the screen
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)

            #craete rectangular coordinates
            x=int(center_x- w/2)
            y=int(center_y- h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(indexes)
reader = Reader(['en'])
font=cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # color=colors[i]
        # print(color)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)
        cv2.putText(img, label, (x, y - 10), font, 3, color=(255, 100, 0), thickness=2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)

        # Crop the numberplate
        crop_image = img[y:y + h, x:x + w]
        # resize the crop size photo
        R_size = cv2.resize(crop_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        # grayscale the image
        gray = cv2.cvtColor(R_size, cv2.COLOR_RGB2GRAY)
        # apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        # thresh the image using otus method
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # create rectangular
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # apply dilate
        dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
        # find the countours
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # histogram equalization

        text=reader.readtext(crop_image)
        print(text[0][1])
        cv2.imshow("crop",crop_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


