"""
Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
accessec on 17 may 2019.


"""
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0], eye[3])

    EAR = (A+B)/(2.0*C)

    return EAR

predictor_path = "D:\college\Alertness_Device\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
print("landmark predictor loaded!!")

EYE_AR_THRESH = 0.26
EYE_AR_CONEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

(lestart,leend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(restart,reend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = VideoStream().start()


while True:
    frame = cap.read()
    frame = imutils.resize(frame,width = 450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        lefteye = shape[lestart:leend]
        righteye = shape[restart:reend]
        leftEAR= eye_aspect_ratio(lefteye)
        rightEAR = eye_aspect_ratio(righteye)

        avg_EAR = (leftEAR+ rightEAR)/2.0

        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)

        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        if avg_EAR <EYE_AR_THRESH:
            COUNTER +=1

        else:
            if COUNTER >= EYE_AR_CONEC_FRAMES:
                TOTAL +=1

            COUNTER = 0


        cv2.putText(frame,"Blinks:{}".format(TOTAL),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,"EAR:{:.2f}".format(avg_EAR),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()
