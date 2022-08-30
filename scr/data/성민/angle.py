import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['front', 'stop']
seq_length = 30
secends = 30

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 각 ab bc의 사이 각 b를 구하는 식
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

created_time = int(time.time())
os.makedirs('angle', exist_ok=True)

cap = cv2.VideoCapture(1)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        for idx, action in enumerate(actions):
            data = []

            ret, img = cap.read()

            img = cv2.flip(img, 1)

            cv2.putText(img, f'{action.upper()}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secends:
                ret, img = cap.read()
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if result.pose_landmarks is not None:
                    for res in result.pose_landmarks.landmark:
                        joint = np.zeros((33, 4))
                        for j, lm in enumerate(result.pose_landmarks.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        landmarks = result.pose_landmarks.landmark

                        footger1 = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]#32
                        footger2 = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]#31
                        foot1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]#28
                        foot2 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]#27
                        knee1 = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]#26
                        knee2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]#25
                        pack1 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #24
                        pack2 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  #23
            
                        angle = calculate_angle(knee1, pack1, pack2)#26 24 23
                        angle1 = calculate_angle(pack1, pack2, knee2)#24 23 25
                        angle2 = calculate_angle(pack1, knee1, foot1)#24 26 28
                        angle3 = calculate_angle(pack2, knee2, foot2)#23 25 27
                        angle4 = calculate_angle(knee1, foot1, footger1)#26 28 32
                        angle5 = calculate_angle(knee2, pack2, footger2)#25 27 31
                    
                        

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label1 = np.array([angle1], dtype=np.float32)
                        angle_label2 = np.array([angle2], dtype=np.float32)
                        angle_label3 = np.array([angle3], dtype=np.float32)
                        angle_label4 = np.array([angle4], dtype=np.float32)
                        angle_label5 = np.array([angle5], dtype=np.float32)
                        angle_label = np.concatenate([angle_label, angle_label1, angle_label2, angle_label3, angle_label4, angle_label5])
                        
                        angle_label = np.append(angle_label, idx)

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)


                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            data = np.array(data)
            print(action, data.shape)
            np.save(os.path.join('angle', f'raw_{action}_{created_time}'), data)

            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('angle', f'seq_{action}_{created_time}'), full_seq_data)
        
        break


