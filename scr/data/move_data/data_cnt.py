import os
import time
import numpy as np
import traceback

list = ["front", "left", "right", "jump", "back", "stay1", "stay2"]
cnt = 0
path = r"2022_AI_PJ\scr\data\move_data\data_collet_school"
error_list = []
for i in list:
    while True:
        try:
            data = np.load(path+f"\{i}\{i}_{cnt}.npy")
            print(path+f"\{i}\{i}_{cnt}.npy", end=",")
            print(np.array(data).shape)
            cnt += 1
        except FileNotFoundError:
            break
        except ValueError:
            print(i, cnt)
            error_list.append([i, cnt])
            traceback.print_exc()
            #time.sleep(2)
            cnt += 1
            pass
            

    cnt = 0
print(error_list)