
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import time




cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()


mpPose=mp.solutions.pose
pose=mpPose.Pose()

mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()

mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh()

# Define the FACE_CONNECTIONS constant
FACE_CONNECTIONS = mpFaceMesh.FACEMESH_TESSELATION

mpDraw=mp.solutions.drawing_utils

Ptime=0
Ctime=0
fps=0

while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    handresults=hands.process(imgRGB)
    poseResults=pose.process(imgRGB)
    faceResults=faceDetection.process(imgRGB)
    
    faceMeshResults=faceMesh.process(imgRGB)

    # if faceMeshResults.multi_face_landmarks:
    #     for faceLms in faceMeshResults.multi_face_landmarks:
    #         mpDraw.draw_landmarks(img,faceLms,FACE_CONNECTIONS)


    if faceResults.detections:
        for id,detection in enumerate(faceResults.detections):
             mpDraw.draw_detection(img,detection)
             print(id,detection)


    if poseResults.pose_landmarks:
        mpDraw.draw_landmarks(img,poseResults.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(poseResults.pose_landmarks.landmark):
            print(id,lm)   

    #print(results.multi_hand_landmarks)

    if handresults.multi_hand_landmarks:
        for handLms in handresults.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==4:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) 
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    Ctime= time.time()
    fps=1/(Ctime-Ptime)
    Ptime=Ctime

    cv2.imshow('Image',img)
     # Wait for a key event
    key = cv2.waitKey(1)

    # Check if the Esc key was pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
