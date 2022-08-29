import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
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
        dump_1.append(np_data)
        dump_2.append(action[action_1])
    sell_data.append(dump_1)
    y_ddata.append(dump_2)

"""print(f"sell_data : {sell_data}, shape : {np.array(sell_data).shape}")
print(f"y_data : {y_ddata}, shape : {np.array(y_ddata).shape}")"""
x_data = np.asanyarray(sell_data)
y_data = np.asanyarray(y_ddata)

#x_data, y_data = np.zeros((16, 30, 33, 3)), np.zeros((30, 1))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state= 42)

x_data = np.asanyarray(x_data)
y_data = np.asanyarray(y_data)
print("x : ", x_data.shape)
print("y : ", y_data.shape, y_data)

model.add(Flatten(input_shape = (22, 30, 33, 4)))
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

model.fit(x_train, y_train, epochs=30)
model.summary()

#model.save(model_path)