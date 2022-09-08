from cProfile import label
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# 데이터를 저장할 파일 경로 안내
DATA_PATH = os.path.join('dataset1') 

# 사용할 액션을 지정
actions = np.array(['front'])

mp_pose = mp.solutions.pose # 홀리스틱 모델을 불러오고
mp_drawing = mp.solutions.drawing_utils # 영상 켰을때 보이는 선이랑 점을 그리기 위한 드로잉 유틸 불러오기

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BRG형식의 이미지를 RGB로 변환
    image.flags.writeable = False                  # 더이상 이미지는 쓰기가 안됨
    results = model.process(image)                 # Make prediction?
    image.flags.writeable = True                   # 이미지 쓰기가 가능해짐
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB형식의 이미지를 RGB로 변환
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def draw_styled_landmarks(image, results):
    #포즈 연결 선 그리기
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(250,250,250), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(250,250,250), thickness=2, circle_radius=2)
                             ) 
#여기는 x, y, z 그리고 보이는 시점의 값
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
#33*4=132
#pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


                    
