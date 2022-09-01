import tensorflow
from keras.models import load_model
import numpy as np

model_path = r"2022_AI_PJ\scr\model_10fps\save_mdel\model_1.h5"
model = load_model(model_path)
data_path = r"2022_AI_PJ\scr\data\move_data\data_collet\front\front_73.npy"
data = np.load(data_path)
dump_1 = []
for x in range(30):
    dump_1.append((data[x]).flatten())
print(np.array(dump_1).shape)
dump_1 = np.expand_dims(dump_1, axis=0)
print(np.array(dump_1).shape)

a = model.predict(dump_1)
print(a)