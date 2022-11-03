from sklearn.model_selection import train_test_split
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
#import numpy as np
import os

path = os.path(r"2022_AI_PJ\scr\model_knn\save_data")

train_x, test_x, train_y, test_y = train_test_split(path, random_state= 42)
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(train_x, train_y)

print(nca_pipe.score(test_x, test_y))