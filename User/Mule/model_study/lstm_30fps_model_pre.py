import cv2
import mediapipe as mp
import keras
import numpy as np
import time
import pyautogui as pag

class ModelPreprocessing():
    def __init__(self, id, path):
        self.id = id
        self.path = path
        self.vital_pose = 0
        self.vital_swich = 1
        self.action_list_1 = ["front", "stay1"]
        self.action_list_2 = ["right", "left", "jump", "stay2", "stay3"]

    def calculate_angle(self, a,b,c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle


    def save_point_body(self, results):
        data = []
        #print("save_point_body_start")
        #print("time : ", time.time())
        
        #if self.vital_pose == 102 or self.vital_pose == 121:#(1, 33, 4)
        if results.pose_landmarks:
            #print("bodys")
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
            #print("d _ shape : ", d.shape)
            
            return np.array(d)
        else:
            #print("test_body")
            data = np.zeros(139)
            return data
    
    def save_point_hand(self, results):
        data = []
        cnt = 0
        #print("time : ", time.time())
        
        #if self.vital_pose == 102 or self.vital_pose == 121:#(1, 33, 4)
        if results.pose_landmarks != None:
            #print("hand")
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
            #print("d _ shape : ", d.shape)
            #np.save(os.path.join(f"2022_AI_PJ\scr\data\move_data\data_test{1}"), d)
            return np.array(d)
        else:
            #print("test_hand")
            data = np.zeros(137)
            return data

        """elif self.vital_pose == 0:
            print("sellect pose")
        else:
            try:
                for i in results.pose_landmarks.landmark:
                    data.append([i.x, i.y, i.z, i.visibility])
                    cnt += 1
                data = np.array(data).flatten()
            except AttributeError:
                data = np.zeros((132))

        return np.array(data)"""

    def pre_data(self):
        model_path = self.path
        self.vital_swich = self.vital_swich
        lock, results = 0, 0
        frame_30_body = []
        frame_30_hand = []
        self.mp_pose = mp.solutions.pose
        #self.mp_hand = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #hand = self.mp_hand.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cam = cv2.VideoCapture(self.id)
        model_1 = keras.models.load_model(model_path+r"\model_body_1fps_stop.h5")
        model_2 = keras.models.load_model(model_path+r"\model_hand_1fps_reset.h5")
        
        while 1:
            
            vital, image = cam.read()
            

            if vital != 1:
                break
            vital_action_f, vital_action_r, vital_action_l, vital_action_j = False, False, False, False
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image)
            

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            

            image = cv2.flip(image, 1)
            cv2.putText(image, f"self.vital_swich : {self.vital_swich}, frame_cnt : {len(frame_30_body)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("test_pose", image)

            key_value = cv2.waitKey(10)#10ms 지연

            if key_value == 32:#spacebar(stop)
                print("key : ", key_value)
                frame_30_body = []#리스트 초기화
                frame_30_hand = []
                self.vital_swich = 0

            elif key_value == 115:#s(save)
                lock = 1
                
            elif key_value == 27:#esc
                print("key : ", key_value)
                break
            elif key_value == 103:#g(go)
                print("key : ", key_value)
                self.vital_swich = 1
            else:
                lock = 0
            
            if self.vital_swich == 1:
                #print("test : ", len(frame_30_body), len(frame_30_hand))
                onefps_data_body = self.save_point_body(results)
                onefps_data_hand = self.save_point_hand(results)
                if type(onefps_data_body) == int:
                    print("data type int. . .")
                    continue
                    
            
                #print("one_fps_data : ", np.array(onefps_data_body).shape)
                #print("flatten() : ", np.array(onefps_data_body))
                
                if len(frame_30_body) == 1:
                    frame_30_body = np.asarray(frame_30_body)
                    frame_30_body = np.expand_dims(frame_30_body, axis = 0)
                    #print("frame_30_body : ", frame_30_body)
                    frame_30_body = frame_30_body[:, :, :-1]
                    pre_1 = model_1.predict(frame_30_body)
                    print(f"front/stay1 : \n{pre_1}")
                    frame_30_body = []

                    #파이오토 1
                    ord_1 = int(np.argmax(pre_1))

                    vital_pose_1 = self.action_list_1[ord_1]

                    print(vital_pose_1)
                    if vital_pose_1 == "front" and vital_action_f == False:
                        pag.keyDown("w")
                        vital_action_f = True
                    elif vital_pose_1 == "front" and vital_action_f == True:
                        pass
                    else:
                        pag.keyUp("w")
                        vital_action_f = False

                    
                    

                if len(frame_30_hand) == 1:
                    frame_30_hand = np.asarray(frame_30_hand)
                    frame_30_hand = np.expand_dims(frame_30_hand, axis = 0)
                    frame_30_hand = frame_30_hand[:, :, :-1]
                    pre_2 = model_2.predict(frame_30_hand)
                    print(f"right/left/jump/stay2 : \n{pre_2}")
                    frame_30_hand = []

                    #파이오토 2
                    ord_2 = int(np.argmax(pre_2))
                    
                    vital_pose_2 = self.action_list_2[ord_2]

                    print(vital_pose_2)
                    if vital_pose_2 == "right" and vital_action_r == False:
                        pag.keyDown("d")
                        vital_action_r = True
                        print("keydown_d")
                    elif vital_pose_2 == "right" and vital_action_r == True:
                        print("keydown_d_pass")
                        pass
                        
                    elif vital_pose_2 == "left" and vital_action_l == False:
                        pag.keyDown("a")
                        vital_action_l = True
                        print("keydown_l")
                    elif vital_pose_2 == "left" and vital_action_l == True:
                        print("keydown_l_pass")
                        pass
                    elif vital_pose_2 == "jump" and vital_action_j == False:
                        pag.keyDown("space")
                        vital_action_l = True
                        print("keydown_j")
                    elif vital_pose_2 == "jump" and vital_action_j == True:
                        print("keydown_j_pass")
                        pass
                    else:
                        print("else _ hand")
                        pag.keyUp("d")
                        pag.keyUp("a")
                        pag.keyUp("space")
                        vital_action_r = False
                        vital_action_l = False
                        vital_action_j = False


                    
                elif lock == 1:
                    lock == 0
                    
                elif len(frame_30_body) > 10:
                    print("frame_cnt error. delete 30fps data")
                    frame_30_body = []

                if len(frame_30_hand) < 10 or len(frame_30_body) < 10:
                    #print("test_data_append")
                    frame_30_body.append(onefps_data_body)
                    frame_30_hand.append(onefps_data_hand)


            #print("time_2", time.time() - ttt)
            #포즈 미실행시 지연 0.03초, 포즈 실행시 지연시간 약 0.06초, 예측 실행시 지연시간 약 0.2초 -> 카메라 프레임이랑 관련 없음, 단순히 반복문 실행시간으로 인하여 그런거임 -> 반복문 안에 반복문 넣어서 프레임 강제 증가? or 몇 프레임 제외 빈 값으로 채워넣어 바로 실행?
            # 약 1초간 ~ > (데이터 수집 / 예측 -> 1fps 수집 약 0.08~0.1초/예측 약 0.1초 )마쳐야 지연이 없어보임


if __name__ == "__main__":
    p = ModelPreprocessing(1, r"2022_AI_PJ\scr\model_10fps\save_mdel")
    p.pre_data()