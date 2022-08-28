import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)
with mp_pose.Pose() as pose:
    while True:
        _, image = camera.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("testpose", cv2.flip(image, 1))


        cnt = []
        if results.pose_landmarks:
            for i in results.pose_landmarks.landmark:
                cnt.append(np.array([i.x, i.y, i.z, i.visibility]))
        else:
            cnt.append(np.zeros(33*4))


        print(np.concatenate([cnt]))
        if cv2.waitKey(10) == 27:
            break