import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

"""
메소드*, 클래스 설명
Dense : 심층신경망에서 마지막 output 값을 softmax로 합칠때, 값을 softmax로 전달하기 전 사용하는 레이어
Sequential : 순차적인 순서라는 뜻, 모델에 레이어를 추가하기 전, 레이어의 바탕이 되는 틀과 같은 존재
Sequential.add(모델명) : 모델을 레이어에 추가
Embedding(임베딩 레이어) : 입력값을 컴퓨터가 인식할 수 있도록 변환해주는 레이어
LSTM(lstm 레이어) : rnn을 응용한 순환신경망 레이어
Dense(dense 레이어) : 

메소드 : 클래스 내에 위치한 함수를 부르는 말
"""
path = r"User\우성민\dataset1\front\0\0.npy"

data = np.load(path)

#x_train, x_test, y_train, y_test = train_test_split(data, random_state=1)

#
model = keras.Sequential()#모델 틀 생성(학습/추론기능 제공)
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
#add < 모델 틀에 레이어 추가
model.add(layers.Embedding(input_dim=1000, output_dim=64))#임베딩 레이어 추가 > 입력값 : 1000, 출력값 : 64

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))#LSTM(n) n = 차원(=히든레이어/은닉층)을 의미

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))#앞 레이어(lstm레이어)의 차원 축소를 위한 레이어

model.summary()#모델 요약(모델의 크기 등 정보 보기)

model.save(r"User\박지영\numpy_study")

model_save = keras.models.load_model(r"User\박지영\numpy_study")