import os

action = {"UTEST" : 0}
path = r"2022_AI_PJ\scr\data\move_data"

list1 = []

for action_l in action:
    for cut_l in range(60, 261+1):
        print(cut_l)
        for frame_l in range(len(os.listdir(path+f"\\{action_l}\\{cut_l}"))):
            list1.append(frame_l)
        if len(list1) == 30:
            list1.clear()
        else:
            print(list1)
            list1.clear()