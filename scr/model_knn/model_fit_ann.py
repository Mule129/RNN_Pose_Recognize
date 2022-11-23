import keras
import numpy as np
from keras import layers
import os
from sklearn.model_selection import train_test_split

y = []
fps_cnt = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.path(r"2022_AI_PJ\scr\model_knn\save_data")
x_train = np.load(r"2022_AI_PJ\scr\model_knn\save_data\train_x.npy")
y_train = np.load(r"2022_AI_PJ\scr\model_knn\save_data\train_y.npy")
"""x_train, x_test, y_train, y_test = train_test_split(data, y, test_size= 0.7, random_state= 1)
x_train = np.array(x_train); x_test = np.array(x_test); y_train = np.array(y_train); y_test = np.array(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
"""
print(x_train.shape, y_train.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(35, 4)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6)
])
model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 100)

model.save(r"2022_AI_PJ\scr\model_knn\save_model\action_6_v1.h5")