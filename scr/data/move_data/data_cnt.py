import os
import numpy as np

path = r"2022_AI_PJ\scr\data\move_data\data_collet"

data = np.load(path+r"\front\front_7.npy")
print(np.array(data).shape)
print(data)
print(type(data))
b = np.asanyarray(data)
print(b)