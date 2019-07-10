import numpy as np
import cv2

def dialogueBox(title, text, width=200, height=130):
    img = np.zeros((height, width, 3), np.uint8)
    img[:,0:width] = (100, 100, 200)
    cv2.putText(img, text, (0, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 2)
    cv2.imshow(title, img)
    cv2.waitKey(0)

dialogueBox("Error", "Bad Thing Happened")
