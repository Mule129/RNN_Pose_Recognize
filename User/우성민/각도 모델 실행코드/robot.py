import cv2
import pyautogui as pag
import mediapipe as mp
import numpy as np
from keras.models import load_model
actions = ['front', 'stop']
seq_length = 30

run=1
#model = load_model(r'model1.h5')
model = load_model(r'2022_AI_PJ\User\우성민\각도 모델 실행코드\model1.h5')

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

#cap = cv2.VideoCapture(2)

cap = cv2.VideoCapture(1)
on_off = "y"
y = input("작동 시작(y/n) : ")#인풋 귀찮으면 앞부분 주석처리시 바로 실행

seq = []
action_seq = []
last_action = None
a=0

while cap.isOpened() and y == "y":
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

            """데이터 정리
            1. all_data > 일반 변수에서 사전형식으로 변경, angle > 바로 집어넣음, 
            2. 큰 데이터들은 사용 후 바로 삭제(메모리에서 삭제는 del, 값만 삭제하고 변수 자체는 그대로 둘시 clear))
            3. 렉 조금 줄었다(뿌듯)"""

            landmarks = result.pose_landmarks.landmark
            all_data = {"footger1" : [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y],
            "footger2" : [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y],
            "foot1" : [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
            "foot2" : [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
            "knee1" : [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            "knee2" : [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            "pack1" : [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
            "pack2" : [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            }

            """
            angle1 = calculate_angle(all_data["pack1"], all_data["pack2"], all_data["knee2"])#24 23 25
            angle2 = calculate_angle(all_data["pack1"], all_data["knee1"], all_data["foot1"])#24 26 28
            angle3 = calculate_angle(all_data["pack2"], all_data["knee2"], all_data["foot2"])#23 25 27
            angle4 = calculate_angle(all_data["knee1"], all_data["foot1"], all_data["footger1"])#26 28 32
            angle5 = calculate_angle(all_data["knee2"], all_data["pack2"], all_data["footger2"])#25 27 31
            """

            angle = [calculate_angle(all_data['knee1'], all_data["pack1"], all_data["pack2"]),
            calculate_angle(all_data["pack1"], all_data["pack2"], all_data["knee2"]), 
            calculate_angle(all_data["pack1"], all_data["knee1"], all_data["foot1"]), 
            calculate_angle(all_data["pack2"], all_data["knee2"], all_data["foot2"]), 
            calculate_angle(all_data["knee1"], all_data["foot1"], all_data["footger1"]), 
            calculate_angle(all_data["knee2"], all_data["pack2"], all_data["footger2"])]

            del all_data
            
            angle = np.concatenate([joint.flatten(), angle])

            seq.append(angle); del angle

            mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            if len(seq) < seq_length:
                continue


            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data)#.squeeze()

            seq.clear(); del input_data
            
            print(y_pred)
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            
            #0 디맨션이어야 하는데 1 디맨션으로 주어져서 지랄
            
            if conf < 0.7:
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
            if run == 1:
                if action == 'front':
                    pag.keyDown('w')
                    pag.keyDown('shift')
                    run = run - 1
            elif run == 0:
                if action == 'front':
                    continue
                elif action == 'stop':
                    run = run + 1
                    pag.keyUp('w')
                    pag.keyUp('shift')

            #action_seq.clear

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):#1ms... 키값 인식하기 힘듦
        break
