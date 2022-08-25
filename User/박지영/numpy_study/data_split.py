from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

path_data = r"2022_AI_PJ\User\우성민\dataset1\front\0\0.npy"
path_name = r"2022_AI_PJ\scr\data\gesture.json"

with open(r"2022_AI_PJ\User\박지영\numpy_study\log.txt", "r") as file:
    d = file.read().split(",")
print(len(d))