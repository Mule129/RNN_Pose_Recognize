import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten, TimeDistributed
from sklearn.model_selection import train_test_split

model = Sequential()
action = {"front" : 0, "stay" : 1}
#"back" : 1, "right" : 2, "left" : 3, "jump" : 4, "stay" : 5}
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

        for x in range(10):
            dump_1.append((np_data[x]).flatten())
            #print(np.array(dump_1).shape)

        sell_data.append(dump_1)
        dump_1 = []
        y_ddata.append(action[action_1])

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
    LSTM(50, activation= "sigmoid", input_shape=(10, 139)),
    Dense(30),
    Dense(10),
    Dense(2, "softmax")])

model.compile(optimizer="adam",
loss="mse", metrics = ["acc"])

model.fit(x_train, y_train, epochs = 35)
model.summary()
a = model.predict(x_test)
#print(a, y_test)
model_path = r"2022_AI_PJ\scr\model_10fps\save_mdel"
model.save(model_path+r"\model_3.h5")