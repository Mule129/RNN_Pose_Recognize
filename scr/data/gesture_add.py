import json

with open(r"2022_AI_PJ\scr\data\gesture.json", "w") as file:
    data = json.load(file)

print(data)