import os

path = r"2022_AI_PJ\scr\data\move_data"
camera_type = ["front", "front_under"]

data, dump = [], []
for i in camera_type:
    dump = list(map(int, os.listdir(f"{path}\{i}")))
    data.append(dump)
    
data[0].sort()
data[1].sort()

for i in range(len(data)):
    for x in data[i]:
        if os.path.isdir(f"{path}\{camera_type[i]}\{x}") != True:
            print(x)

#print(data)
