from imp import load_module
import numpy as np

from keras.models import load_model

model_path = r"2022_AI_PJ\scr\model_10fps\save_mdel\model_body.h5"

body_model = load_module(model_path)

#body_model.predict()

