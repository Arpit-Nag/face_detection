# -*- coding: utf-8 -*-
"""
Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
accessed on 17 may 2021.
"""


from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2
import simpleaudio



face_predictor = "D:\college\Alertness_Device\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
print("loading camera....")
video = VideoStream().start()

wave_obj = simpleaudio.WaveObject.from_wave_file("sound.wav")

print("loading....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor)

time.sleep(2.0)
take = 0
fps=FPS().start()
while True:
    take+=1
    frame = video.read()
    frame = imutils.resize(frame,width = 1080)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)


    if len(rects)==0:
        # print(f"NO FACE DETECTED!! in loop{take}")
        play_obj = wave_obj.play()
        play_obj.wait_done()


    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        for(x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,0,255),-1)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break
    fps.update()

fps.stop()
print(f"elasped time :{fps.elapsed()}")
print(f"approximate FPS :{fps.fps()} ")
cv2.destroyAllWindows()
video.stop()
