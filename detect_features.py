# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:36:49 2020
Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
accessec on 17 may 2019.


@author: Hp
"""


import numpy as np
import dlib
import cv2
import imutils
from imutils import face_utils

predictor_path =  "D:\college\Alertness_Device\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
image_path = "D:\college\\Alertness_Device\\faces\\face_1.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
print(image_path)
image = cv2.imread(image_path)
image = imutils.resize(image,width = 500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rects = detector(gray,1)

for(i,rect) in enumerate(rects):
    shape = predictor(gray,rect)
    shape = face_utils.shape_to_np(shape)

    for(name,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        clone = image.copy()
        cv2.putText(clone,name,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        for(x,y) in shape[i:j]:
            cv2.circle(clone,(x,y),1,(0,0,255),-1)

        (x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y+h,x:x+w]
        roi = imutils.resize(roi,width = 250,inter = cv2.INTER_CUBIC)
        cv2.imshow("ROI",roi)
        cv2.imshow("image",clone)
        cv2.waitKey(0)


    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("image",output)
    cv2.waitKey(0)
