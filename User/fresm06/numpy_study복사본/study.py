import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

"""
메소드*, 클래스 설명
Dense : 심층신경망에서 마지막 output 값을 softmax로 합칠때, 값을 softmax로 전달하기 전 사용하는 레이어
Sequential : 순차적인 순서라는 뜻, 모델에 레이어를 추가하기 전, 레이어의 바탕이 되는 틀과 같은 존재. 모델추가가 가장 간편하지만 복잡하게 다루기엔 어려움이 있음
funcional : sequential와 같은 역할을 하지만 레이어를 추가할때 입력/ 출력값 개수 등의 옵션을 직접 지정해줘야함(input, output)
subclassing : pytorch와 유사한 방식의 레이어 틀, 직접 모델 클래스를 생성하고 상속받아야함, __init__에는 사용할 레이어 생성, call함수에서 생성한 레이어 가져와줘야함

Sequential.add(모델명) : sequential 틀에 (모델명) 레이어 추가
Embedding(임베딩 레이어) : 입력값을 컴퓨터가 인식할 수 있도록 변환해주는 레이어
LSTM(lstm 레이어) : rnn을 응용한 순환신경망 레이어
Dense(dense 레이어) : 

메소드 : 클래스 내에 위치한 함수를 부르는 말
"""
path_0 = r"User\우성민\dataset1\front\0\0.npy"
path_1 = r"2022_AI_PJ\User\우성민\dataset1\front\0\0.npy"

data = np.load(path_1)
print(data)

#x_train, x_test, y_train, y_test = train_test_split(data, random_state=1)

#
model = keras.Sequential()#모델 틀 생성(학습/추론기능 제공)

# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
#add < 모델 틀에 레이어 추가
model.add(layers.Embedding(input_dim=1000, output_dim=64))#임베딩 레이어 추가 > 입력값 : 1000, 출력값 : 64

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128, activation = "relu"))#LSTM(n) n = 차원(=히든레이어/은닉층)을 의미

# Add a Dense layer with 10 units.
model.add(layers.Dense(10, activation = "softmax"))#앞 레이어(lstm레이어)의 차원 축소를 위한 레이어


model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["categorical_accuracy"])

model.fit(x_data, y_data, epochs = 10, callbacks = ["tb_callback"])

model.summary()#모델 요약(모델의 크기 등 정보 보기)

model.save(r"2022_AI_PJ\User\박지영\numpy_study\model_save")

model_save = keras.models.load_model(r"2022_AI_PJ\User\박지영\numpy_study\model_save")