import os
import numpy as np
list = ["front", "left", "right", "jump", "back"]
i = 0
data = np.load(r"2022_AI_PJ\scr\data\move_data\data_collet_school\right\right_32.npy")
print(np.array(data).shape)
print(data[0])