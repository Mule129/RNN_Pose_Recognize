import cv2
import mediapipe as mp
import numpy as np
import os

class PoseCollet():#함수 종료시 자원회수 되는지 확인하기
    def __init__(self, id, path):
        self.id = id
        self.vital_pose = 0
        self.data_path = path
        self.mp_pose = mp.solutions.pose
        self.mp_hand = mp.solutions.hands

    def calculate_angle(self, a,b,c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 

    def save_point(self, results):
        data = []

        if self.vital_pose == 102 or self.vital_pose == 121:#(1, 33, 4)
            if results.pose_landmarks:
                for i in results.pose_landmarks.landmark:
                    data.append([i.x, i.y, i.z, i.visibility])

                landmarks = results.pose_landmarks.landmark
                data = np.array(data)
                footger1 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]#32
                footger2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]#31
                foot1 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]#28
                foot2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]#27
                knee1 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]#26
                knee2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]#25
                pack1 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y] #24
                pack2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]  #23
    
                angle = self.calculate_angle(knee1, pack1, pack2)#26 24 23
                angle1 = self.calculate_angle(pack1, pack2, knee2)#24 23 25
                angle2 = self.calculate_angle(pack1, knee1, foot1)#24 26 28
                angle3 = self.calculate_angle(pack2, knee2, foot2)#23 25 27
                angle4 = self.calculate_angle(knee1, foot1, footger1)#26 28 32
                angle5 = self.calculate_angle(knee2, pack2, footger2)#25 27 31
            
                

                angle_label = np.array([angle], dtype=np.float32)
                angle_label1 = np.array([angle1], dtype=np.float32)
                angle_label2 = np.array([angle2], dtype=np.float32)
                angle_label3 = np.array([angle3], dtype=np.float32)
                angle_label4 = np.array([angle4], dtype=np.float32)
                angle_label5 = np.array([angle5], dtype=np.float32)
                angle_label = np.concatenate([angle_label, angle_label1, angle_label2, angle_label3, angle_label4, angle_label5])

                angle_label = np.append(angle_label, self.vital_pose)
                
                #angle_label = np.append(angle_label)
                d = np.concatenate([data.flatten(), angle_label])
                # print(d.shape)
                #np.save(os.path.join(f"2022_AI_PJ\scr\data\move_data\data_test{1}"), d)
                return np.array(d)
            else:
                data = np.zeros(139)
                return data
            #print(f"pose_cont : {cnt}, {np.array(data).shape}")
        elif self.vital_pose == 0:
            print("sellect pose")
            """else:
                if results.pose_landmarks:
                    for i in results.pose_landmarks.landmark:
                        data.append([i.x, i.y, i.z, i.visibility])
                    landmarks = results.pose_landmarks.landmark
                    data = np.array(data)
                    p12 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]# 12
                    p11 = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]#11
                    p14 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]#14
                    p13 = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]#13
                    p16 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]#16
                    p15 = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]#15
                    p24 = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y] #24
                    p23 = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]  #23

                    angle = self.calculate_angle(p24, p12, p14)
                    angle1 = self.calculate_angle(p23, p11, p13)
                    angle2 = self.calculate_angle(p12, p14, p16)
                    angle3 = self.calculate_angle(p11, p13, p15)
                
                    

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label1 = np.array([angle1], dtype=np.float32)
                    angle_label2 = np.array([angle2], dtype=np.float32)
                    angle_label3 = np.array([angle3], dtype=np.float32)
                    angle_label = np.concatenate([angle_label, angle_label1, angle_label2, angle_label3])
                    
                    angle_label = np.append(angle_label, self.vital_pose)
                    
                    #angle_label = np.append(angle_label)
                    d = np.concatenate([data.flatten(), angle_label])
                    #print(d.shape)
                    #np.save(os.path.join(f"2022_AI_PJ\scr\data\move_data\data_test{1}"), d)
                    return np.array(d)
                else:
                    data = np.zeros(137)"""
        else:
            if results.multi_hand_landmarks != None:
                for tow_hand in results.multi_hand_landmarks:
                    for i in self.mp_hand.HandLandmark:
                        dump = tow_hand.landmark[i]
                        
                        if np.array(data).shape != (42, 3):
                            pass
                        else:
                            data.append([dump.x, dump.y, dump.z])#(x, y, z)
                        #print(np.array(data).shape)
                        
                return data

            else:
                data = np.zeros(126)
                return data
            # 양손 : 42(가끔 한손을 두손으로 인식할 때가 있음), 한손 : 21, 손 x : 0 
            # -> 최대 손 인식 1로 변경
            # """

        #np.array().flatten : 다차원 배열 -> 1차원 배열로 변환
        #return np.array(data)
    
    def save_pose(self, data, pose):
        """
        pose : {1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98), u : stop_2{hand}, y : stop1{bodys}}
        """
        str_pose = ""
        if pose == 0:#python switch - dictionary방식 참고
            print("sellect pose")
            return
        elif pose == 102:
            str_pose = "front"
        elif pose == 114:
            str_pose = "right"
        elif pose == 108:
            str_pose = "left"
        elif pose == 106:
            str_pose = "jump"
        elif pose == 98:
            str_pose = "back"
        elif pose == 121:#y
            str_pose = "stay1"
        elif pose == 117:
            str_pose = "stay2"
        elif pose == 105:
            str_pose = "stay3"

        try:
            add_path = f"{self.data_path}\{str_pose}"            
            np.save(add_path+f"\{str_pose}_{len(os.listdir(add_path))}", data)
            print(add_path)
        except FileNotFoundError:
            print("make folder")
            os.makedirs(add_path)
            np.save(add_path+f"\{str_pose}_{len(os.listdir(add_path))}", data)
        

    def cap_pose(self):
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        cam = cv2.VideoCapture(self.id)
        hand = self.mp_hand.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #{0 : save_off, 1 : save_on}
        vital_swich = 0
        #{1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98)}
        self.vital_pose = 0
        frame_30 = []
        #save lock{0 : off, 1 : save}
        lock = 0
        while True:
            vital, image = cam.read()

            if vital != 1:
                print("camera not open")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.vital_pose == 102 or self.vital_pose == 121: 
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

            else:
                results = hand.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, 
                        self.mp_hand.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                except TypeError:
                    #print("skip 1fps(hand not found)")
                    pass
 
            image = cv2.flip(image, 1)
            cv2.putText(image, f"vital_swich : {vital_swich}, self.vital_pose : {self.vital_pose}, frame_cnt : {len(frame_30)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("test_pose", image)
            
            key_value = cv2.waitKey(10)#10ms 지연
            
            if key_value == 32:#spacebar(stop)
                print("key : ", key_value)
                frame_30 = []#리스트 초기화
                vital_swich = 0

            elif key_value == 115:#s(save)
                lock = 1
                
            elif key_value == 27:#esc
                print("key : ", key_value)
                break
            elif key_value == 103:#g(go)
                print("key : ", key_value)
                vital_swich = 1
            else:
                lock = 0
            
            if vital_swich == 0 and (key_value == 114 #right
            or key_value == 102 #front
            or key_value == 108 #left
            or key_value == 106 #jump
            or key_value == 98 #back
            or key_value == 121 #y - stop(bodys)
            or key_value == 117 or key_value == 105): #u - stop(hand)
                self.vital_pose = key_value

            if vital_swich == 1:
                onefps_data = self.save_point(results)
            
                #print("one_fps_data : ", np.array(onefps_data).shape)
                #print("flatten() : ", np.array(onefps_data))
                
                if len(frame_30) == 30:
                    self.save_pose(frame_30, self.vital_pose)
                    frame_30 = []
                elif lock == 1:
                    lock == 0#초기화 안되는 문제 해결하기
                    self.save_pose(frame_30, self.vital_pose)
                    
                elif len(frame_30) > 30:
                    print("frame_cnt error. delete 30fps data")
                    frame_30 = []
                else:
                    frame_30.append(onefps_data)
                        

if __name__ == "__main__":
    p = PoseCollet(0, r"2022_AI_PJ\scr\data\move_data\data_collet_small_pose")
    p.cap_pose()
