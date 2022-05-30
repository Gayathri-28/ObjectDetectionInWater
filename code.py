import numpy as np
import cv2
import os
net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
CLASSES=[]
with open("coco.names","r")as f:
    CLASSES=[line.strip() for line in f.readlines()]
print(CLASSES)
layer_names=net.getLayerNames()
outputlayers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]

img=cv2.imread("sample9.jpg")
img=cv2.resize(img,(512,512),fx=0.4,fy=0.4)
height,width,channels=img.shape


def object_detection(img):
    blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs=net.forward(outputlayers)
    boxes=[]
    class_ids=[]
    confidences=[]
    # for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
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
                x=int(center_x -w/2)
                y=int(center_y -h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
    font=cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x,y,w,h=boxes[i]
        label=str(CLASSES[class_ids[i]])
    
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,512,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
object_detection(img)
