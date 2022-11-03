import mediapipe as mp
import numpy as np
import cv2
import os
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DataCollet():
    def __init__(self):
        self.id = 0
    
    def calculate_angle(self, a,b,c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 

    def cam_show(self, pose_list, fps_cnt):
        pose = mp.solutions.pose
        mp_pose = pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp.solutions.hands
        mp_hand = hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        drawing = mp.solutions.drawing_utils
        drawing_s = mp.solutions.drawing_styles

        collat_data, collat_lable = [], []
        cnt = 0; loop = 0
        cam = cv2.VideoCapture(self.id)

        model = keras.models.load_model(r"2022_AI_PJ\scr\model_knn\save_model\action_6.h5")

        while True:
            vital, img = cam.read()
            if vital != True:
                self.id += 1
                print("camera not connect. . port change")
                return 1
            
            if cv2.waitKey(1) == 27:
                break

            img.flags.writeable = False

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result_pose = mp_pose.process(img)
            result_hand = mp_hand.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"pose : {pose_list[cnt]}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, f"pose : {pose_list[cnt]}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if result_hand.multi_hand_landmarks and result_pose.pose_landmarks:
                drawing.draw_landmarks(img, result_pose.pose_landmarks, 
                pose.POSE_CONNECTIONS,
                landmark_drawing_spec = drawing_s.get_default_pose_landmarks_style())

                for hand_landmarks in result_hand.multi_hand_landmarks:
                    drawing.draw_landmarks(img, hand_landmarks, 
                    hands.HAND_CONNECTIONS,
                    drawing_s.get_default_hand_landmarks_style(),
                    drawing_s.get_default_hand_connections_style())

                ##pose##
                data = []
                for i in result_pose.pose_landmarks.landmark:
                    data.append([i.x, i.y, i.z, i.visibility])
                result_pose = result_pose.pose_landmarks.landmark

                pp = pose.PoseLandmark
                
                #12 - RIGHT_SHOLDER (y) #28 - RIGHT_ANKLE (y)
                tall_p = result_pose[pp.RIGHT_ANKLE.value].y - result_pose[pp.RIGHT_SHOULDER.value].y
                #15 - RIGHT_WRIST(y) #16 - LEFT_WRIST(y)
                hand_v = abs(result_pose[pp.RIGHT_WRIST.value].y - result_pose[pp.LEFT_WRIST.value].y)
                #15 - 24(RIGHT_HIP(y))
                hand_hr = result_pose[pp.RIGHT_WRIST.value].y - result_pose[pp.RIGHT_HIP.value].y
                #16 - 23(LEFT_HIP(y))
                hand_hl = result_pose[pp.LEFT_WRIST.value].y - result_pose[pp.LEFT_HIP.value].y
                #28(RIGHT_ANKLE) - 27(LEFT_ANKLE)
                #24, 26, 28 // 23, 25, 27 (angle)
                ra_28 = [result_pose[pp.RIGHT_ANKLE.value].x, result_pose[pp.RIGHT_ANKLE.value].y]
                ra_26 = [result_pose[pp.RIGHT_KNEE.value].x, result_pose[pp.RIGHT_KNEE.value].y]
                ra_24 = [result_pose[pp.RIGHT_HIP.value].x, result_pose[pp.RIGHT_KNEE.value].y]

                ra_27 = [result_pose[pp.LEFT_ANKLE.value].x, result_pose[pp.LEFT_ANKLE.value].y]
                ra_25 = [result_pose[pp.LEFT_KNEE.value].x, result_pose[pp.LEFT_KNEE.value].y]
                ra_23 = [result_pose[pp.LEFT_HIP.value].x, result_pose[pp.LEFT_HIP.value].y]

                angle_r = self.calculate_angle(ra_28, ra_26, ra_24)
                angle_l = self.calculate_angle(ra_27, ra_25, ra_23)

                ##hand##
                #data_hand.append(result_hand.multi_hand_landmarks)
                #print(result_hand[hh.WRIST.value].x)
                result_hand = result_hand.multi_hand_landmarks[0].landmark#손은 멀티라 2차원임, 리스트 하나 까줘야함
                hh = hands.HandLandmark

                #tall : 0(WRIST), 12(MID_FINGER_TIP)(x only)
                tall_h1 = abs(result_hand[hh.WRIST.value].x - result_hand[hh.MIDDLE_FINGER_TIP.value].x)
                #result : 11(MID_FINGER_DIP) - 9(MID_FINGER_MCP)(x only)
                tall_h2 = abs(result_hand[hh.MIDDLE_FINGER_DIP.value].x - result_hand[hh.MIDDLE_FINGER_MCP.value].x)
                #angle : 4(THUMB_TIP), 2(THUMB_MCP), 5(INDEX_FINGER_MCP)
                ra_27 = [result_hand[hh.THUMB_TIP.value].x, result_hand[hh.THUMB_TIP.value].y]
                ra_25 = [result_hand[hh.THUMB_MCP.value].x, result_hand[hh.THUMB_MCP.value].y]
                ra_23 = [result_hand[hh.INDEX_FINGER_MCP.value].x, result_hand[hh.INDEX_FINGER_MCP.value].y]

                angle_h = self.calculate_angle(ra_28, ra_26, ra_24)
                #print(np.array(data).shape)
                #data_pose.append([[hand_hr, hand_hl, angle_r, angle_l], [hand_v, tall_h1, tall_h2, angle_h], data])
                data.append([hand_hr, hand_hl, angle_r, angle_l])
                data.append([hand_v, tall_h1, tall_h2, angle_h])
                data = np.array([data])
                result_pre = model.predict(data)
                print(result_pre)
                
            else:
                pass

            cv2.imshow("DataCollet", img)
                



if __name__ == "__main__":
    dc = DataCollet()
    start = dc.cam_show(pose_list=["work", "jump", "left", "right", "stop", "attack"], fps_cnt=100)
    if start == 1:
        dc.cam_show()
    