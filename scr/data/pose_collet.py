import cv2
import mediapipe as mp
import numpy as np
import os

class PoseCollet():
    def __init__(self, id, pose, path):
        self.id = id
        self.pose = pose
        #저장위치 입력받거나, 자동인식으로 수정)
        self.data_path = r"2022_AI_PJ\scr\data\move_data\data_collet"

    def save_point(results):
        data = []

        if results.pose_landmarks:#(1, 33, 4)
            for i in results.pose_landmarks.landmark:
                data.append([i.x, i.y, i.z, i.visibility])
        else:
            data.append(np.zeros(33*4))

        #np.array().flatten : 다차원 배열 -> 1차원 배열로 변환
        return np.array(data)
    
    def save_pose(self, data, pose):
        """
        pose : {1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98)}
        45 line > AttributeError: 'int' object has no attribute 'data_path' error 해결하기
        add_path os.listdir(path) 파일 개수가 0일때 에러 안뜨는지 확인
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
        add_path = f"{self.data_path}\{str_pose}\{len(os.listdir(self.path))}"
        np.save(add_path, data)

        return
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    cam = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        #{0 : save_off, 1 : save_on}
        vital_swich = 0
        #{1 : front(102), 2 : right(114), 3 : left(108), 4 : jump(106), 5 : back(98)}
        vital_pose = 0
        frame_30 = []

        while True:
            vital, image = cam.read()

            if vital != 1:
                print("camera not open")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.flip(image, 1)
            cv2.putText(image, f"vital_swich : {vital_swich}, vital_pose : {vital_pose}, frame_cnt : {len(frame_30)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("test_pose", image)
            

            key_value = cv2.waitKey(10)#지연 10ms

            if key_value == 115:#s(stop)
                print("key : ", key_value)
                vital_swich = 0
            elif key_value == 27:#esc
                print("key : ", key_value)
                break
            elif key_value == 103:#g(go)
                print("key : ", key_value)
                vital_swich = 1
            
            #조건문 수정(현재 102만 작동됨(가장 앞에것). 일일히 쓰기는 코드 길이낭비(사실 일일히 쓰기 귀찮음))
            if vital_swich == 0 and key_value == (102 or 114 or 108 or 106 or 98):
                vital_pose = key_value

            if vital_swich == 1:
                onefps_data = save_point(results)
            
                #print("one_fps_data : ", np.array(onefps_data).shape)
                #print("flatten() : ", np.array(onefps_data))
                
                if len(frame_30) == 30:
                    save_pose(0, frame_30, vital_pose)
                    frame_30 = []
                elif len(frame_30) > 30:
                    print("frame_cnt error")
                    frame_30 = []
                else:
                    frame_30.append(onefps_data)
            
