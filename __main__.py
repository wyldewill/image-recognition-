#Import Modules
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
import random
import numpy as np

#Parse Args
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", default="mobilenet_ssd_v2/coco_labels.txt", help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.35, help="minimum probability to filter weak detections")
ap.add_argument("-w", "--width", type=int, default=700, help="width of frame")
args = vars(ap.parse_args())

#Dialog Box Func
def dialogBox(title, text, width=200, height=130):
    img = np.zeros((height, width, 3), np.uint8)
    img[:,0:width] = (100, 100, 200)
    cv2.putText(img, text, (0, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.imshow(title, img)
    cv2.waitKey(0)

#Initialise
print("[INFO] parsing class labels...")
labels = {}

for row in open(args["labels"]):
        # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()

print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

print("[INFO] starting video stream...")

#Only Run If Camera Is Active
try:
        vs = VideoStream(src=0)
        testVar = imutils.resize(vs.start().read(), width=args["width"])
except Exception as e:
        dialogBox("Error", str(e), width=1000)
        quit()
else:
    vs = vs.start()

#Control Panel
def updateThreshold(x):
        args["confidence"] = x/100

def updateSize(x):
    args["width"] = x if x > 100 else 100

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", 'Controls', int(args['confidence']*100), 100,  updateThreshold)
cv2.createTrackbar("Size", 'Controls', int(args['width']),  1000,  updateSize)

#Draw Func
def drawFrame():

    thresh = args['confidence']
    
    frame = vs.read()
    frame = imutils.resize(frame, width=args["width"])
    frame = cv2.flip(frame, 1)
    orig = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    results = model.DetectWithImage(frame, threshold=thresh,
    keep_aspect_ratio=True, relative_coord=False)

    for r in results:
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        label = labels[r.label_id]

        colour = (0, 255*r.score, 255*(1-r.score))
        cv2.rectangle(orig, (startX, startY), (endX, endY), colour, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = f"{label}: {int(r.score * 100)}%"
        cv2.putText(orig, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    cv2.putText(orig, f"{int(thresh*100)}% Threshold", (args["width"]-130, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255*thresh, 255*(1-thresh)), 2)

    cv2.imshow("Image Recognition", orig)

def close():
    cv2.destroyAllWindows()
    vs.stop
    dialogBox("Quit", "Goodbye!", width=300)
    exit()

#loop
looping = True

while looping:
    drawFrame()

    key = cv2.waitKey(1)

    if key == ord("q"):
        looping = False
        close()
