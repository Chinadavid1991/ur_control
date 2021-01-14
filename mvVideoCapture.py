import numpy as np
import cv2
from threading import Lock, Thread
import time


class VideoCapture(cv2.VideoCapture, Thread):
    def __init__(self, video=0):
        cv2.VideoCapture.__init__(self, video)
        Thread.__init__(self)
        self.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.set(cv2.CAP_PROP_SATURATION, 65)
        self.set(cv2.CAP_PROP_EXPOSURE, 120)
        # self.set(cv2.CAP_PROP_GAMMA, 0.5)
        self.set(cv2.CAP_PROP_CONTRAST, 150)
        self.__lock = Lock()
        self.setDaemon(True)
        self.start()

    def run(self):
        while True:
            self.__lock.acquire()
            super(VideoCapture, self).grab()
            self.__lock.release()
            time.sleep(0.01)

    def read(self, image=None):
        self.__lock.acquire()
        f, frame = super(VideoCapture, self).read()
        self.__lock.release()
        # frame = cv2.flip(frame, flipCode=-1)
        return f, frame

    def stop(self):
        super(VideoCapture, self).release()
