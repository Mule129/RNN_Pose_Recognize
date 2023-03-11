from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import numpy as np
import os
from sklearn.model_selection import train_test_split

path_1 = r"2022_AI_PJ\User\박지영\model_study\save_mdel"
path_2 = r"2022_AI_PJ\scr\data\move_data"
model = Sequential()
model = load_model(path_1)

x_data, y_data = [], []
dump_1, dump_2 = [], []
action = {"front_under" : 0, "front" : 1}

for action_l in action:
    for cut_l in range(len(os.listdir(path_2+f"\{action_l}"))):
        #print(cut_l)
        dump_1, dump_2 = [], []
        for frame_l in range(len(os.listdir(path_2+f"\{action_l}\{cut_l}"))):
            frame_data = np.load(path_2+f"\{action_l}\{cut_l}\{frame_l}.npy")
            dump_1.append(frame_data)
            dump_2.append(action[action_l])
        x_data.append(dump_1)
        y_data.append(dump_2)
        
 
print(np.array(x_data).shape)
print(np.array(y_data).shape)
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
print(type(x_data), type(y_data))
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 42)

print(model.predict(x_test))
model.summary()
_, test_acc = model.evaluate(x_test, y_test, verbose = 2)
print("테스트 정확도 : ", test_acc)