import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


action = {"right": 0, "left": 1, "jump": 2, "stay3": 3}
# "right" : 0, "left" : 1, "jump" : 2, "stay2" : 3, "stay3" : 4
#

x_data, y_data = [], []
dump_1, dump_2 = [], []

path_1 = r"scr\data\move_data\data_collet_small_pose"

sell_data = []
y_ddata = []
cnt = 0
for action_1 in action:
    for i in range(len(os.listdir(f"{path_1}\{action_1}"))):
        cnt += 1
        path_2 = f"{path_1}\{action_1}\{action_1}_{i}.npy"
        np_data = np.load(path_2)

        for x in range(1):
            print(dump_1)
            print(np.array(dump_1).shape)
            dump_1.append((np_data[x]).flatten())
            # print(np.array(dump_1).shape)

        sell_data.append(dump_1)
        dump_1 = []
        y_ddata.append(action[action_1])

print(np.array(y_ddata).shape)
print(np.array(sell_data).shape)

x_data = np.asanyarray(sell_data)
y_data = y_ddata
y_data = np.asanyarray(y_data)

y_data = to_categorical(y_data, len(action))
# x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_data = x_data[:, :, :-1]

print("x : ", x_data.shape, "\ny : ", y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential([
    LSTM(100, activation="sigmoid", input_shape=(5, 136)),
    Dense(32),
    Dropout(0.3),
    Dense(16),
    Dense(8),
    Dense(4),
    Dropout(0.3),
    Dense(len(action), "softmax")])

model.compile(optimizer="adam",
              loss="mse", metrics=["acc"])

model.fit(x_train, y_train, epochs=300)
model.summary()
a = model.predict(x_test)
# print(a, y_test)
model_path = r"2022_AI_PJ\scr\model_10fps\save_mdel"
model.save(model_path + r"\model_hand_small.h5")
