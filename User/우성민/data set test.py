import cv2
import mediapipe as mp
import numpy as np
#from tensorflow.python.keras.models import load_model
import time, os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#시퀀스 길이와, 작동 시간 설정
seq_length = 30
secs_for_action = 30

#제스처(몸짓)종류 지정
actions=['walk', 'left', 'right', 'back', 'jump']


created_time = int(time.time())
os.makedirs('데이터세트', exist_ok=True)
#카메라를 가져오고, 미디어파이프에서 설정값 설정해 주기
#그리고 카메라 불러오는걸 성공했는지 안했는지를 여부로 컨티뉴하거나 반복하기
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#캠이 열려있는 동안 
while cap.isOpened():
    #idx, action 이 actions 에 존재 한다면
    for idx, action in enumerate(actions):

        data=[]

        #ret, img 에 카매라를 읽은걸 저장해 주고
        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'준비하는동안 기다려 주세요 {action.upper()} 액션을 위한...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        #시작 시간을 현재 시간으로 설정해 준다
        start_time=time.time()
        #현재 시간을 시작 시간으로 뺐을 때 처음에 지정해준 숫자보다 작을 경우
        while time.time() - start_time < secs_for_action:
            #이제 몸짓을 데이터 화 시키는 작업을 업로드 할 예정
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            #이미지를 RGB형식으로 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = pose.process(img)
            #이미지를 BRG형식으로 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if result.POSE_LANDMARKS is not None:
                for res in result.POSE_LANDMARKS:
                    joint = np.zeros((33, 6))#np.zeros는 0으로 가득한 함수 생성


                ################################################################


        #시퀀스 데이터를 저장 하는 과정
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

