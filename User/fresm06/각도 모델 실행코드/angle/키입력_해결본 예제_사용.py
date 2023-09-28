import pyautogui as pag
import cv2
import mediapipe as mp
import numpy as np
run=1
jump=1
a=0
max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
                 
# MediaPipe hands models
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition models
file = np.genfromtxt('gesture_train_fy.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            
            # Draw gesture result
            if idx in gesture.keys():
                cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
            # Movement of hamster
            if run == 1:
                if gesture[idx] == 'fist':
                    pag.keyDown('space')
                    run = run - 1
            elif run == 0:
                if gesture[idx] == 'fist':
                    exit
                else:
                    run = run + 1
                    pag.keyUp('space')


            """if gesture[idx]=='fist':
                h.wheels(120, 120)
                wait(1)
            if gesture[idx]=='five':
                h.wheels(0, 0)
                wait(1)
            if gesture[idx]=='yeah':
                h.wheels(-120, 120)
                wait(1)
            if gesture[idx]=='one':
                h.wheels(120, -120)
                wait(1)
            if gesture[idx]=='three':
                h.wheels(-120, -120)
                wait(1)
            
            if gesture[idx]=='spiderman':
                a=a+2
                while a > 0:
                    h.wheels(-120, 120)
                    wait(300)
                    h.wheels(120, -120)
                    wait(300)
                    a=a-1"""
        
            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)



    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
