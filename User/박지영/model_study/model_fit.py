import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
action = {"front_under" : 0, "front" : 1}
path = r"2022_AI_PJ\scr\data\move_data"

x_data, y_data = [], []
dump_1, dump_2 = [], []
 
for action_l in action:
    for cut_l in range(len(os.listdir(path+f"\{action_l}"))):
        #print(cut_l)
        dump_1 = []
        for frame_l in range(len(os.listdir(path+f"\{action_l}\{cut_l}"))):
            frame_data = np.load(path+f"\{action_l}\{cut_l}\{frame_l}.npy")
            dump_1.append(frame_data)
            dump_2.append(action[action_l])
        x_data.append(dump_1)
        y_data.append(dump_2)
        
 
print(np.array(x_data).shape)
print(np.array(y_data).shape)
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
print(type(x_data), type(y_data))
 
model.add(LSTM(16, 
               input_shape = (30, 132), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))
 
model_path = r"2022_AI_PJ\User\박지영\numpy_study\model_save"
model.compile(loss='mean_squared_error', optimizer='adam')
 
history = model.fit(x_data, y_data, 
                    epochs=200, 
                    batch_size=32)


model.save(r"2022_AI_PJ\User\박지영\model_study\save_mdel")