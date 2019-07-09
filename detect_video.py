from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
import random

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", default="mobilenet_ssd_v2/coco_labels.txt", help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.35, help="minimum probability to filter weak detections")
ap.add_argument("-w", "--width", type=int, default=700, help="width of frame")
args = vars(ap.parse_args())

print("[INFO] parsing class labels...")
labels = {}

for row in open(args["labels"]):
        # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()

print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

def updateThreshold(x):
        args["confidence"] = x/100

cv2.namedWindow("uielements", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", 'uielements', int(args['confidence']*100), 100, updateThreshold)

def drawFrame():
        frame = vs.read()
        frame = imutils.resize(frame, width=args["width"])
        frame = cv2.flip(frame, 1)
        orig = frame.copy()

        # prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        start = time.time()
        results = model.DetectWithImage(frame, threshold=args["confidence"],
                keep_aspect_ratio=True, relative_coord=False)
        end = time.time()

        # loop over the results
        for r in results:
                # extract the bounding box and box and predicted class label
                box = r.bounding_box.flatten().astype("int")
                (startX, startY, endX, endY) = box
                label = labels[r.label_id]

                # draw the bounding box and label on the image
                colour = (0, 255*r.score, 255*(1-r.score))
                cv2.rectangle(orig, (startX, startY), (endX, endY), colour, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                text = f"{label}: {int(r.score * 100)}%"
                cv2.putText(orig, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                cv2.putText(orig, f"{int(args['confidence']*100)}% Threshold", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 50, 256), 2)


        #update the window
        cv2.imshow("Image Recognition", orig)
	
#draw loop
looping = True

while looping:
        drawFrame()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                print("[INFO] quitting")
                looping = False

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
print("[INFO] stopped")