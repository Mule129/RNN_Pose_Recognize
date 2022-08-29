from cProfile import label
#from tensorflow.python.keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard

# 각 액션마다 30개의 영상을 저장할 예정을 밝힘
no_sequences = 30

#각 영상은30프레임 사용
sequence_length = 30

# 데이터를 저장할 파일 경로 안내
DATA_PATH = os.path.join('dataset1') 

# 사용할 액션을 지정
actions = np.array(['front'])

mp_pose = mp.solutions.pose # 홀리스틱 모델을 불러오고
mp_drawing = mp.solutions.drawing_utils # 영상 켰을때 보이는 선이랑 점을 그리기 위한 드로잉 유틸 불러오기

#로그라는 폴더 만들기 아마 뉴런 제작 예정 폴더 일듯
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

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
#시퀀스와 라벨이 비어있는 리스트 생성
sequences, labels = [], []
#액션 안에 액션스 를 저장
for action in actions:
    for sequence in range(no_sequences):
        window = []
        #append 는 리스트 안에 변수 추가하는 명령어
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
#LSTM활용해서 렐루활성 함수를 사용해 RNN으로 예측 한다.

X = np.array(sequences)

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, stratify = y)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, 0.2, 0.1]

actions[np.argmax(res)]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
model.summary()
res = model.predict(X_test)

actions[np.argmax(res[4])]
actions[np.argmax(y_test[4])]

model.save('action.h5')