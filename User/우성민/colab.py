import numpy as np
import cv2
import mediapipe as mp
from tensorflow.python.keras.models import load_model
from keras.models import Sequential

load_model('model.h5')

actions = np.array(['front_under'])
model = Sequential()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

def color(img, model):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    result = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img, result

def randmark(img, result):
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def line(img, result):
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2))   

def track(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

seq = []
pred = []
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        R, frame = cap.read()
        img, result = color(frame, pose)
        line(img, result)
        re = track(result)
        seq.append(re)
        seq = seq[-30:]

        if len(seq) == 30:
            res = model.predict(np.expand_dims(seq, axis=0))[0]
            print(actions[np.argmax(res)])
            pred.append(np.argmax(res))
        #print(pred)
        cv2.imshow("testpose", cv2.flip(img, 1))