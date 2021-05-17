"""
Adrian Rosebrock,Blink Detection Using Open CV,
https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
accessed on 17 may 2021.


"""

import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import time
import simpleaudio
from threading import Thread
import tkinter as tk

switch = True
root = tk.Tk()
wave_obj = simpleaudio.WaveObject.from_wave_file("sound.wav")
play_obj = wave_obj.play
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = WebcamVideoStream(0).start()

def scanning():
    scanning.eyes = True

    def frames():
        fps = FPS().start()
        while switch:
            frame = vs.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(
                gray
            )
            if len(eyes)==0:
                scanning.eyes =  True
            else:
                scanning.eyes =  False

            for (x, y, w, h) in eyes:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def face_det():
        while switch:
            if scanning.eyes:
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
