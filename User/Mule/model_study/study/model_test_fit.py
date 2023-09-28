import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
#from tensorboard import

gesture_list = np.array([0])

path = r"2022_AI_PJ\scr\data\move_data\front"
data, labels = [], []
for i in gesture_list:
    for j in range(1, len(os.listdir(path))+1):
        window = []
        for z in range(len(os.listdir(path+f"\{j}"))):
            dump = np.load(path+f"\\{j}\\{z}.npy")
            window.append(dump)
        #print(f"{window}\n{j}")
        #print(len(window))
        data.append(window)
        labels.append(gesture_list[0])

print(np.array(data).shape)
print(np.array(labels).shape)

#print(len(data), len(labels))
y = to_categorical(131).astype(int)
print(f"{np.array(y).shape} : y ")

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(gesture_list.shape[0], activation='softmax'))

res = [.2]

gesture_list[np.argmax(res)]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(data, y, epochs=2000)

#models.summary()

model.save(r"2022_AI_PJ\User\박지영\numpy_study\model_save")