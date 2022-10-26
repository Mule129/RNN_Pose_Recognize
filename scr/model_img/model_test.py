import time
import cv2
import mediapipe
import numpy as np
"""data = np.random.rand(4000, 100)
full_data = []
for i in range(4000-30):#data[1:31] 데이터 아웃 예방
    full_data.append(data[i:i + 30])
print(data[1:1+30])#data[1:31]
print(np.array(full_data).shape)#1~30프레임씩 잘라서 새로운 배열에 넣어줌"""
cam = cv2.VideoCapture(0)


cnt = 0
while True:
    _, img = cam.read()
    cv2.imshow("test", img)
    cnt += 1
    if cnt > 30:
        break
    
    if cv2.waitKey(10) == 27:
        break

cnt = 0
t1 = time.time()
while True:
    _, img = cam.read()
    cv2.imshow("test", img)
    cnt += 1
    
    if time.time() - t1 > 15:
        break
    if cv2.waitKey(10) == 27:
        break
print(time.time() - t1, cnt)

cnt = 0
t2 = time.time()
while True:
    while time.time() - t2 < 15:
        _, img = cam.read()
        cv2.imshow("test", img)
        cnt += 1
        if cv2.waitKey(10) == 27:
            break
    break
    
print(time.time() - t2, cnt)

cnt = 0
t3 = time.time()
while time.time() - t3 < 15:
    _, img = cam.read()
    cv2.imshow("test", img)
    cnt += 1
    if cv2.waitKey(10) == 27:
        break
print(time.time() - t3, cnt)