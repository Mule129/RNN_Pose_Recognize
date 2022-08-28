import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split#3차원을 나눌 수 없음 -> 직접 함수 만들기

model = Sequential()
action = {"front_under" : 0, "front" : 1}
path = r"2022_AI_PJ\scr\data\move_data"

x_data, y_data = [], []
dump_1, dump_2 = [], []
 
for action_l in action:
    
    for cut_l in range(len(os.listdir(path+f"\{action_l}"))):
        #print(cut_l)
        for frame_l in range(len(os.listdir(path+f"\{action_l}\{cut_l}"))):
            frame_data = np.load(path+f"\{action_l}\{cut_l}\{frame_l}.npy")
            dump_1.append(frame_data)
            dump_2.append(action[action_l])
            
        x_data.append(dump_1)
        y_data.append(dump_2)
        dump_1, dump_2 = [], []

def chack_data_len(data):
    x_1 = data[0]
    #print(x_1, "\n", len(x_1))
    x_2 = data[1]
    x_3 = data[2]
    return len(x_1), len(x_2), len(x_3)

print(chack_data_len(x_data))
print(type(x_data), type(y_data))
print(len(x_data), len(y_data))

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
print(np.array(x_data).shape)
print(np.array(y_data).shape)
print(type(x_data), type(y_data))

def data_split(x_data, y_data):
    np.array(x_data[0])
    np.array(y_data[0])
    return #x_train, x_test, y_train, y_test

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 1)

#print("type_x_train, x_test, y_train, y_test : ", type(x_train), type(x_test), type(y_train), type(y_test))
#print("shape : ", np.array(x_train).shape , np.array(x_test).shape, np.array(y_train).shape, np.array(y_test).shape)

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

#model.predict(x_test)
model.save(model_path)