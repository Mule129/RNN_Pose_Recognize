import cv2
import mediapipe as mp
import keras
import numpy as np

class ModelPreprocessing():
    def __init__(self, id, path):
        self.id = id
        self.path = path
        
        self.vital_swich = False


    def save_point(self, results):
        data, dump = [], []
        cnt = 0

        
        if results.pose_landmarks.landmark:
            for i in results.pose_landmarks.landmark:
                data.append([i.x, i.y, i.z, i.visibility])
                cnt += 1
        else:
            data = np.zeros((33, 4))
        print(data, type(data), np.array(data).shape)
        return np.array(data)

    def pre_data(self):
        model_path = self.path+r"\model_1.h5"
        vital_swich = self.vital_swich
        lock = 0
        frame_30, fps_data = [], []

        mp_pose = mp.solutions.pose
        mp_hand = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hand = mp_hand.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cam = cv2.VideoCapture(self.id)
        model = keras.models.load_model(model_path)
        fps_cnt = 0
        
        while 1:
            vital, image = cam.read()

            if vital != 1:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.vital_swich > 0: 
                results = pose.process(image)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
            else:
                results = hand.process(image)

                try:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, 
                        mp_hand.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                except TypeError:
                    #print("skip 1fps(hand not found)")
                    pass

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = cv2.flip(image, 1)
            cv2.putText(image, f"vital_swich : {vital_swich}, frame_cnt : {len(frame_30)}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
            
            if vital_swich == 1:
                onefps_data = self.save_point(results)
            
                #print("one_fps_data : ", np.array(onefps_data).shape)
                #print("flatten() : ", np.array(onefps_data))
                
                if len(frame_30) == 30:
                    print(frame_30)
                    frame_30 = np.asarray(frame_30)
                    pre = model.predict(frame_30)
                    print(pre)
                    frame_30 = []

                elif lock == 1:
                    lock == 0#초기화 안되는 문제 해결하기
                    
                elif len(frame_30) > 30:
                    print("frame_cnt error. delete 30fps data")
                    frame_30 = []

                else:
                    print("test")
                    frame_30.append(onefps_data)


if __name__ == "__main__":
    p = ModelPreprocessing(0, r"2022_AI_PJ\User\박지영\model_study\save_mdel")
    p.pre_data()
