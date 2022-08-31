import cv2
import mediapipe as mp
import keras
import numpy as np
import time
import pyautogui

class ModelPreprocessing():
    def __init__(self, id, path):
        self.id = id
        self.path = path
        
        self.vital_swich = False


    def save_point(self, results):
        data = []
        cnt = 0
        #print("time : ", time.time())
        
        try:
            for i in results.pose_landmarks.landmark:
                data.append([i.x, i.y, i.z, i.visibility])
                cnt += 1
            data = np.array(data).flatten()
        except AttributeError:
            data = np.zeros((33 * 4))

        return np.array(data)

    def pre_data(self):
        model_path = self.path
        self.vital_swich = self.vital_swich
        lock, results = 0, 0
        frame_30 = []

        self.mp_pose = mp.solutions.pose
        #self.mp_hand = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #hand = self.mp_hand.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cam = cv2.VideoCapture(self.id)
        model_1 = keras.models.load_model(model_path+r"\model_1.h5")
        model_2 = keras.models.load_model(model_path+r"\model_2.h5")
        fps_cnt = 0
        
        while 1:
            #ttt = time.time()
            vital, image = cam.read()
            #rint("time_1", time.time() - ttt)

            if vital != 1:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image)
            #results_2 = hand.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            
            """try:
                for hand_landmarks in results_2.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, 
                    self.mp_hand.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            except TypeError:
                #print("skip 1fps(hand not found)")
                pass"""

            

            image = cv2.flip(image, 1)
            cv2.putText(image, f"self.vital_swich : {self.vital_swich}, frame_cnt : {len(frame_30)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("test_pose", image)

            key_value = cv2.waitKey(10)#10ms 지연

            if key_value == 32:#spacebar(stop)
                print("key : ", key_value)
                frame_30 = []#리스트 초기화
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
                onefps_data = self.save_point(results)
                if type(onefps_data) == int:
                    continue
                    
            
                #print("one_fps_data : ", np.array(onefps_data).shape)
                #print("flatten() : ", np.array(onefps_data))
                
                if len(frame_30) == 10:
                    frame_30 = np.asarray(frame_30)
                    frame_30 = np.expand_dims(frame_30, axis = 0)
                    pre_1 = model_1.predict(frame_30)
                    pre_2 = model_2.predict(frame_30)
                    print(f"front/stay : {pre_1}, right/left/jump/stay : {pre_2}")
                    frame_30 = []
                    
                    #파이오토 실행
                    """if pre_1[0] >= 0.8:
                        print("w")"""

                elif lock == 1:
                    lock == 0
                    
                elif len(frame_30) > 10:
                    print("frame_cnt error. delete 30fps data")
                    frame_30 = []

                else:
                    frame_30.append(onefps_data)
            #print("time_2", time.time() - ttt)
            #포즈 미실행시 지연 0.03초, 포즈 실행시 지연시간 약 0.06초, 예측 실행시 지연시간 약 0.2초 -> 카메라 프레임이랑 관련 없음, 단순히 반복문 실행시간으로 인하여 그런거임 -> 반복문 안에 반복문 넣어서 프레임 강제 증가? or 몇 프레임 제외 빈 값으로 채워넣어 바로 실행?
            # 약 1초간 ~ > (데이터 수집 / 예측 -> 1fps 수집 약 0.08~0.1초/예측 약 0.1초 )마쳐야 지연이 없어보임


if __name__ == "__main__":
    p = ModelPreprocessing(1, r"2022_AI_PJ\User\박지영\model_study\save_mdel")
    p.pre_data()