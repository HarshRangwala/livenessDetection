# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:41:18 2020

@author: Anuj
"""

from scipy.spatial import distance as dist
#from imutils.video import FileStream
from imutils.video import VideoStream 
from imutils.video import FPS
from imutils import face_utils 
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-sh","--shape-predictor", required = True,
                help = "Path to facial landmark shape detector")
args = vars(ap.parse_args())

# Function to calculate the EAR
def eye_aspect_ratio(eye):
    '''
    

    Parameters
    ----------
    eye : TYPE
        DESCRIPTION.
    
    A : compute the eucilidean distances between two
        sets of vertical eye landmarks (x, y)- coordinates
    
    C : Compute the eucilidean distance between the 
        horizontal eye landmark (x, y)- coordinate
        
    ear: Compute the eye aspect ratio
    
    Returns ear
    -------
    None.

    '''
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    C = dist.euclidean(eye[0], eye[3])
    
    ear = (A+B)/(2.0*C)
    
    return ear

EYE_AR_THRESH = 0.21
RATIO_THRESH = 0.0017
EYE_AR_CONSEC_FRAME = 3

COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(leftst, lefted) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightst, righted) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src = 0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    
    frame = vs.read()
    frame =  imutils.resize(frame, width = 400)
    
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(grayScale, 0)
    
    for rect in rects:
        '''
        shape : determine the facial landmarks for the face region
                then convert the facial landmark (x, y)-coordinates
                to a NumPy array
        
        lefteye : left eye coordinates
        
        righteye : right eye coordinates
        
        leftEAR  : eye aspect ratio for left ear
        
        rightEAR : eye aspect ratio for right ear
        
        ear      : average eye aspect ratio
        
        '''
        shape = predictor(grayScale, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[leftst:lefted]
        rightEye = shape[rightst:righted]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR)/2.0
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        ear_ratio = ear/w
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # If the eye aspect ratio is below the blink ratio,
        # then increment the counter
        if ear_ratio < RATIO_THRESH:
            COUNTER += 1
            print("======EAR======")
            print(ear)
            print("===============")
            print(w)
            print("===============")
            print(ear_ratio)
        # otherwise, the eye aspect ratio is not be
        # below the blink threshold
        else:
            # if the eyes were closed for a sufficient
            # number of frames then increment the total 
            # number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAME:
                TOTAL += 1
            
            COUNTER = 0
            
        # draw the total number of blinks on the frame 
        # along with the computed eye aspect ratio
        # for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.5f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
            
        
        
        