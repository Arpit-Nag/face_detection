# -*- coding: utf-8 -*-
"""
Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
accessed on 17 may 2021.

Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
accessed on 17 may 2021.


"""
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2
import simpleaudio
from threading import Thread
import tkinter as tk

switch = True
root = tk.Tk()

face_predictor = "D:\college\Alertness_Device\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor)
wave_obj = simpleaudio.WaveObject.from_wave_file("sound.wav")
play_obj = wave_obj.play
vs = WebcamVideoStream(0).start()

def scanning():

    scanning.face = True
    def frames():
        fps = FPS().start()
        while switch:
            frame =vs.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            rects = detector(gray,0)
            if len(rects)==0:
                scanning.face =  True
            else:
                scanning.face =  False
            fps.update()
        fps.stop()
        print(f"elasped time :{fps.elapsed()}")
        print(f"approximate FPS :{fps.fps()} ")
        cv2.destroyAllWindows()
        vs.stop()

    def face_det():
        while switch:
            if scanning.face:
                play_obj()
                play_obj().wait_done()
            else:
                continue

    thread1=Thread(target=frames)
    thread2 = Thread(target=face_det)
    thread1.start()
    thread2.start()

def switchon():
    global switch
    switch = True
    print('switch on')
    scanning()

def switchoff():
    print('switch off')
    global switch
    switch = False

def kill():
    root.destroy()

onbutton = tk.Button(root, text = "scan", command = switchon)
onbutton.pack()
offbutton =  tk.Button(root, text = "OFF", command = switchoff)
offbutton.pack()
killbutton = tk.Button(root, text = "EXIT", command = kill)
killbutton.pack()

root.mainloop()
