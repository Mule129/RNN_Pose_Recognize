import cv2
import mediapipe
import numpy
import typing

class DataCollection:
    def __init__(self, pose_type : list):
        self.pose_type = pose_type
    def camera(self, camera_number = 0, learning_data_count = 30):
        cv2.VideoCapture(camera_number)