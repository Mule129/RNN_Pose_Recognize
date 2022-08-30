import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten, TimeDistributed
from sklearn.model_selection import train_test_split

model = Sequential()
action = {"front" : 0, "back" : 1, "right" : 2, "left" : 3, "jump" : 4}
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
        sell_data.append(dump_1)
        dump_1 = []
        y_ddata.append(action[action_1])

#print(y_ddata)
print(np.array(y_ddata).shape)
print(np.array(sell_data).shape)
x_data = np.asanyarray(sell_data)
y_data = y_ddata
y_data = np.asanyarray(y_data)
print("x : ", x_data.shape)
print("y : ", y_data.shape)

from keras.utils import to_categorical
y_data = to_categorical(y_data, len(action))
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state= 42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


model = Sequential([
    LSTM(50, activation = "sigmoid", input_shape=(30, 132)),
    Dense(10),
    Dense(5, activation="softmax")])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=30, batch_size=10, validation_data = (x_test, y_test))
model.summary()
model_path = r"2022_AI_PJ\User\박지영\model_study\save_mdel"
model.save(model_path)
prd = model.predict(x_test)
print(prd)
print(y_test)