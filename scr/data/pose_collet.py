import cv2
import mediapipe as mp
import numpy as np
import os

class PoseCollet():#함수 종료시 자원회수 되는지 확인하기
    def __init__(self, id, path):
        self.id = id
        self.vital_pose = 0
        #저장위치 입력받거나, 자동인식으로 수정)
        self.data_path = path
        self.mp_hand = 0

    def save_point(self, results):
        data = []

        if self.vital_pose <= 102:#(1, 33, 4)
            if results.pose_landmarks:
                for i in results.pose_landmarks.landmark:
                    data.append([i.x, i.y, i.z, i.visibility])
            else:
                data.append(np.zeros(33*4))

        elif self.vital_pose == 0:
            print("sellect pose")
        else:
            if results.multi_hand_landmarks:
                for i in results.multi_hand_landmarks:
                    for x in self.mp_hand.HandLandmark:
                        t = i.landmark[x]
                        print(t, type(t), "test??")
                        data.append([t.x, t.y, t.z])#(x, y, z)
                    """print("hand_pose_save", np.array(i).shape)
                    data.append(i.landmark.x, i.landmark.y, i.landmark)"""
            else:
                data.append(np.zeros(33*3))#x, y, z


        #np.array().flatten : 다차원 배열 -> 1차원 배열로 변환
        return np.array(data)
    
    def save_pose(self, data, pose):
        """
        pose : {1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98)}
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

        try:
            add_path = f"{self.data_path}\{str_pose}"            
            np.save(add_path+f"\{str_pose}_{len(os.listdir(add_path))}", data)
            print(add_path)
        except FileNotFoundError:
            print("make folder")
            os.makedirs(add_path)
            np.save(add_path+f"\{str_pose}_{len(os.listdir(add_path))}", data)
        

    def cap_pose(self):
        mp_pose = mp.solutions.pose
        self.mp_hand = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        cam = cv2.VideoCapture(self.id)
        hand = self.mp_hand.Hands(static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #{0 : save_off, 1 : save_on}
        vital_swich = 0
        #{1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98)}
        self.vital_pose = 0
        frame_30 = []

        while True:
            vital, image = cam.read()

            if vital != 1:
                print("camera not open")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.vital_pose <= 102: 
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
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
                    print("skip 1fps(hand not found)")
                    pass
 
            image = cv2.flip(image, 1)
            cv2.putText(image, f"vital_swich : {vital_swich}, self.vital_pose : {self.vital_pose}, frame_cnt : {len(frame_30)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("test_pose", image)
            

            key_value = cv2.waitKey(10)#지연 10ms

            if key_value == 115:#s(stop)
                print("key : ", key_value)
                frame_30 = []#리스트 초기화
                vital_swich = 0
            elif key_value == 27:#esc, 파일 삭제
                print("key : ", key_value)
                break
            elif key_value == 103:#g(go)
                print("key : ", key_value)
                vital_swich = 1
            
            #조건문 수정(현재 102만 작동됨(가장 앞에것). 일일히 쓰기는 코드 길이낭비(사실 일일히 쓰기 귀찮음))
            if vital_swich == 0 and key_value == (114 or 102 or 108 or 106 or 98):
                self.vital_pose = key_value

            if vital_swich == 1:
                onefps_data = self.save_point(results)
            
                #print("one_fps_data : ", np.array(onefps_data).shape)
                #print("flatten() : ", np.array(onefps_data))
                
                if len(frame_30) == 30:
                    self.save_pose(frame_30, self.vital_pose)
                    frame_30 = []
                elif len(frame_30) > 30:
                    print("frame_cnt error")
                    frame_30 = []
                else:
                    frame_30.append(onefps_data)
                        

if __name__ == "__main__":
    p = PoseCollet(0, r"2022_AI_PJ\scr\data\move_data\data_collet")
    p.cap_pose()
