import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
#from keras.preprocessing.sequence import pad_sequences

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
        dump_2.append(action_1)
    sell_data.append(dump_1)
    y_ddata.append(dump_2)

print(f"sell_data : {sell_data}, shape : {np.array(sell_data).shape}")
print(f"y_data : {y_ddata}, shape : {np.array(y_ddata).shape}")
x_data = np.asanyarray(sell_data)
y_data = np.asanyarray(y_ddata)
print(x_data, y_data)


model.add(LSTM(16,
               input_shape = (10, 5), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(2, activation="softmax"))

model_path = r"2022_AI_PJ\scr\data\model_process\model_save"
model.compile(loss='mean_squared_error', optimizer='adam')
 
history = model.fit(x_data, y_data, 
                    epochs=200, 
                    batch_size=32)


model.save(model_path)