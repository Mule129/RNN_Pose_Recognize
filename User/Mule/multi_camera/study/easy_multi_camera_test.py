from atexit import register
from unittest import result
import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

cap_0 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    _, image_0 = cap_0.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image_0 = pose
    image_0.flags.writeable = False
    image.flags.writeable = False
    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resulte_0 = pose.process(image_0)
    results = pose.process(image)


    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_0.flags.writeable = True
    image_0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image_0,
        resulte_0.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow("test2", cv2.flip(image_0, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cap_0.release()