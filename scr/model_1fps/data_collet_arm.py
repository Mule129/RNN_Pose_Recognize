import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ["right"]
seq_length = 30
secends = 15

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
os.makedirs('angle1', exist_ok=True)
cnt = 0
cap = cv2.VideoCapture(0)
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
            if cv2.waitKey(10) == 27:
                break
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secends:
                ret, img = cap.read()
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cnt += 1
                print(len(result.pose_landmarks.landmark))
                if result.pose_landmarks is not None:
                    for res in result.pose_landmarks.landmark:
                        cnt += 1
                        joint = np.zeros((33, 4))
                        for j, lm in enumerate(result.pose_landmarks.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        landmarks = result.pose_landmarks.landmark

                        p12 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]# 12
                        p11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]#11
                        p14 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]#14
                        p13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]#13
                        p16 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]#16
                        p15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]#15
                        p24 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #24
                        p23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  #23
            
                        angle = calculate_angle(p24, p12, p14)
                        angle1 = calculate_angle(p23, p11, p13)
                        angle2 = calculate_angle(p12, p14, p16)
                        angle3 = calculate_angle(p11, p13, p15)
                    
                        

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label1 = np.array([angle1], dtype=np.float32)
                        angle_label2 = np.array([angle2], dtype=np.float32)
                        angle_label3 = np.array([angle3], dtype=np.float32)
                        angle_label = np.concatenate([angle_label, angle_label1, angle_label2, angle_label3])
                        
                        angle_label = np.append(angle_label, idx)
                        #print(f"joint_1 : {np.array(joint).shape}")
                        d = np.concatenate([joint.flatten(), angle_label])
                        #print(f"joint_2 : {np.array(d).shape}")

                        data.append(d)

                        mp_drawing.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

                """cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break"""
            print("cnt : ", cnt)
            data = np.array(data)
            print(action, data.shape)
            #np.save(os.path.join('angle1', f'raw_{action}_{created_time}'), data)
            #print(f"len_data : {len(data)}")
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])
                #print(f"data seq len : {len(data[seq:seq + seq_length])}")

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            #np.save(os.path.join('angle1', f'seq_{action}_{created_time}'), full_seq_data)
        
        break


