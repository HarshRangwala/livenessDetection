import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type = str, required = True, help = "Path to the input video")
ap.add_argument("-o", "--output", type = str, required = True, help = "path to the output directory with cropped faces")
ap.add_argument("-d", "--detector", type = str, required = True, help= "Path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "Minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type = int, default = 16, help = "# Number of faces to skip before applying face detection")
args  vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# Loop that process over the frames
while True:

    (grabbed, frame) = vs.read()

    # If the frame is not grabbed it means
    # we have reached the end of the stream
    if not steam:
        break

    # Increament the read count
    read += 1

    # see if we should process this frame
    if read % args["skip"] != 0:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        # Grab the highest probability face detection index
        i = np.argmax(detections[0, 0, : 2])
        confidence = detections[0, 0, i, 2])

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            p = os.path.sep.join([args["output"], "{}.png".format(saved)])

            cv2.imwrite(p, face)

            saved += 1
            
            print("[INFO] saved {} to disk".format(p))
            
vs.release()
cv2.destroyAllWindows()

