import cv2
import pyautogui as pag
import mediapipe as mp
import numpy as np
from keras.models import load_model
actions = ['right', 'left', 'jump']
seq_length = 30
left=1
right=1
model = load_model(r'2022_AI_PJ\scr\model_1fps\model_save\model3.h5')
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture(0)
seq = []
action_seq = []
last_action = None
a=0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if result.pose_landmarks is not None:
        for res in result.pose_landmarks.landmark:
            joint = np.zeros((33, 4))
            for j, lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            landmarks = result.pose_landmarks.landmark
            p12 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]# 12
            p11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]#11
            p14 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]#14
            p13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]#13
            p16 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]#16
            p15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]#15
            p24 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #24
            p23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  #23

            angle = calculate_angle(p24, p12, p14)
            angle1 = calculate_angle(p23, p11, p13)
            angle2 = calculate_angle(p12, p14, p16)
            angle3 = calculate_angle(p11, p13, p15)
            angle = [angle, angle1, angle2, angle3]

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)
            mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            if len(seq) < seq_length:
                continue
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            #print(y_pred)
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            seq.clear()
            #0 디맨션이어야 하는데 1 디맨션으로 주어져서 지랄
            
            if conf < 0.9 :
                continue
            action = actions[i_pred]
            #action_seq.append(action)
            """if len(action_seq) < 3:
                print('test')
                continue
            else:
                print("?")"""
            """this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:"""
            if action == 'jump':
                pag.press('space')

            if right == 1:
                if action == 'right':
                    pag.keyDown('d')
                    right = right - 1
            elif right == 0:
                if action == 'right':
                    continue
                else:
                    right = right + 1
                    pag.keyUp('d')

            if left == 1:
                if action == 'left':
                    pag.keyDown('a')
                    left = left - 1
            elif left == 0:
                if action == 'left':
                    continue
                else:
                    left = left + 1
                    pag.keyUp('a')
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break