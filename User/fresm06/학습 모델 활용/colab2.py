import cv2
import mediapipe as mp
import numpy as np
from keras.models import Sequential
from tensorflow.python.keras.models import load_model

load_model('model1.h5')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
model = Sequential()
actions = np.array(['front_under'])

# def color(img, models):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img.flags.writeable = False
#     result = models.process(img)
#     img.flags.writeable = True
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     return img, result

# def randmark(img, result):
#     mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def line(img, result):
    mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2))   

def track(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

sequence = []
predictions = []
cnt = []
action_seq = []
camera = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while True:
        _, image = camera.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        line(image, results)


        if results.pose_landmarks:
            for i in results.pose_landmarks.landmark:
                cv2.imshow("testpose", cv2.flip(image, 1))
                cnt.append(np.array([i.x, i.y, i.z, i.visibility]))
                res = model.predict(cnt)
                if len(res) < 30:
                    continue
                else:
                    
                    input_data = np.expand_dims(np.array(sequence[-30:], dtype=np.float32), axis=0)
                    y_pred = model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    if conf < 0.9:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action) 

                    if len(action_seq) < 3:
                        continue 
        else:
            cv2.imshow("testpose", cv2.flip(image, 1))
            cnt.clear()
            cnt.append(np.zeros(33*4))
        

        print(np.concatenate([cnt]))
        if cv2.waitKey(10) == 27:
            break
            
