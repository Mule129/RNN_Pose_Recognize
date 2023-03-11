import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import tensorflow as tf
import keras
from keras import layers
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = tensorflow.keras.Sequential()
action = {"front_under" : 0}
path = r"2022_AI_PJ\scr\data\move_data"

x_data, y_data = [], []
dump_1, dump_2 = [], []
for action_l in action:
    for cut_l in range(1, len(os.listdir(path+f"\\{action_l}"))):
        #print(cut_l)
        dump_1 = []
        for frame_l in range(len(os.listdir(path+f"\\{action_l}\\{cut_l}"))):
            frame_data = np.load(path+f"\\{action_l}\\{cut_l}\\{frame_l}.npy")
            dump_1.append(frame_data)
            dump_2.append(action[action_l])
        x_data.append(dump_1)
        y_data.append(dump_2)
        

print(np.array(x_data).shape)
print(np.array(y_data).shape)
#x_data = np.asarray(x_data)
#y_data = np.asarray(y_data)
x_data = np.expand_dims(x_data, -1)
y_data = np.expand_dims(y_data, -1)
#y_data = to_categorical(y_data)
#print(y_data)

print(np.array(x_data).shape)
print(np.array(y_data).shape)
print(type(x_data), type(y_data))
#print(x_data.dtype)
"""X = np.asarray(x_data).astype(np.float32)
Y = np.asarray(y_data).astype(np.float32)"""

model.add(LSTM(16, 
               input_shape = (30, 132), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))

model_path = r"2022_AI_PJ\User\우성민\numpy_study\model_save"
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_data, y_data, 
                    epochs=200, 
                    batch_size=16, 
                    callbacks=[early_stop, checkpoint])