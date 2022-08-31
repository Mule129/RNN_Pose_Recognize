import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten, TimeDistributed
from sklearn.model_selection import train_test_split

model = Sequential()
action = {"front" : 0, "back" : 1}
path = r"2022_AI_PJ\scr\data\move_data\data_collet"

x_data, y_data = [], []
dump_1, dump_2 = [], []

path_1 = f"2022_AI_PJ\scr\data\move_data\data_collet"
sell_data = []
y_ddata = []
for action_1 in action:
    for i in range(len(os.listdir(f"{path_1}\{action_1}"))):
        path_2 = f"2022_AI_PJ\scr\data\move_data\data_collet\{action_1}\{action_1}_{i}.npy"
        np_data = np.load(path_2)
        for x in range(30):
            dump_1.append((np_data[x]).flatten())
            print(np.array(dump_1).shape)
        sell_data.append(dump_1)
        dump_1 = []
        """for x in range(len(np_data[0])):
            np_data = np.array(np_data[x]).flatten()"""
        #dump_1.append(np_data)
        y_ddata.append(action[action_1])
    
    #y_ddata.append(dump_2)
    #dump_2 = []
print(y_ddata)
print(np.array(y_ddata).shape)
"""print(f"sell_data : {sell_data}, shape : {np.array(sell_data).shape}")
print(f"y_data : {y_ddata}, shape : {np.array(y_ddata).shape}")"""
print(np.array(sell_data).shape)
x_data = np.asanyarray(sell_data)
y_data = y_ddata
y_data = np.asanyarray(y_data)
print("x : ", x_data.shape)
print("y : ", y_data.shape)

from keras.utils import to_categorical
y_data = to_categorical(y_data, 2)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state= 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = Sequential([
    LSTM(20, return_sequences=True, input_shape=(30, 132)),
    LSTM(20, return_sequences=True), 
    TimeDistributed(Dense(1))])

model.summary()

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

model.fit(x_train, y_train, epochs=30)
model.summary()

#model.save(model_path)