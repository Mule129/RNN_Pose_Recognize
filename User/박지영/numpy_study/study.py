import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras

"""
단어 설명
Dense : 심층신경망에서 마지막 output 값을 softmax로 합칠때, 값을 softmax로 전달하기 전 사용하는 레이어

"""
path = r"2022_AI_PJ\User\우성민\dataset1\front\0\1.npy"

data = np.load(path)

#x_train, x_test, y_train, y_test = train_test_split(data, random_state=1)

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()