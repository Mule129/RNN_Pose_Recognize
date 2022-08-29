import os
import numpy as np
list = ["front", "left", "right", "jump", "back"]
i = 0
data = np.load(f"2022_AI_PJ\scr\data\move_data\data_collet\{list[i]}\{list[i]}_10.npy")
print(np.array(data).shape)
print(data[0])