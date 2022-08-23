import multiprocessing
import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pag
file = np.genfromtxt('C:/Users/user/Desktop/Rock-Paper-Scissors-Machine-c1d1a59071b68eb3ff4a059150dbcae164c20107/data/gesture_train_fy.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)
ran=1
jump=1
a=0
max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
#mediapipe AttributeError 해결하기
#line 40, 79 멀티프로세싱으로 생기는 오류는 x, 지역변수나 접근권한과 관련있는듯
#-> 해결(08.23) with mp_pose.Pose as pose: 파일 닫고 작업해서 생긴문제
#멀티프로세싱으로 함수 실행위치가 어긋남 -> 한개의 함수만 실행됨
#-> 해결(08.23) id(지역변수)->self.id(클래스 인스턴스)로 변경

class multiCamera(multiprocessing.Process):
    def __init__(self, cameraName, id):
        multiprocessing.Process.__init__(self)
        self.cameraName = cameraName
        self.id = id

    def run(self):
        print(f"{self.cameraName} starting, usb port id : {self.id}")
        if self.id == 0:
            print("body_pose def start")
            hand_pose(self.cameraName, self.id)
        else:
            print("hand_pose def start")
            #body_pose(self.cameraName, self.id)

def hand_pose(cameraName, id):#python 데코레이터(@) 참고하기
    mp_hand = mp.solutions.hands
    mp_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(id)

    print(f"{cameraName} open vital : {camera.isOpened()}")
    
    if camera.isOpened():
        vital, image = camera.read()
    else:
        vital = False



    with mp_hand.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hand:

        while vital:
            vital, image = camera.read()

            image.flags.writeable = False

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
            )
            if results.multi_hand_landmarks is not None:

                ges_result = []
                for res in results.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])
            
            # Draw gesture result
                    if idx in gesture.keys():
                        cv2.putText(image, text=gesture[idx].upper(), org=(int(res.landmark[0].x * image.shape[1]), int(res.landmark[0].y * image.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                ges_result.append({
                    'ges': gesture[idx],
                    'id': id
                })
#####################################################################################
            cv2.imshow(cameraName, cv2.flip(image, 1))
            if ran == 1:
                if ges_result[1]['ges'] == 'fist':
                    pag.keyDown('j')
                    pag.keyDown('k')
                    ran = ran - 1
            elif ran == 0:
                if ges_result[1]['ges'] == 'fist':
                    exit
                else:
                    ran = ran + 1
                    pag.keyUp('j')
                    pag.keyUp('k')
            
            if jump == 1:
                if ges_result[2]['ges'] == 'fist':
                    pag.keyDown('b')
                    jump = jump - 1
            elif jump == 0:
                if ges_result[2]['ges'] == 'fist':
                    exit
                else:
                    jump = jump + 1
                    pag.keyUp('b')
            if cv2.waitKey(20) == 27:
                print(f"{cameraName}_off : Hand")
                break

    camera.release()

def body_pose(cameraName, id):
    mp_pose = mp.solutions.pose
    mp_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(id)

    print(f"Camera open vital : {camera.isOpened()}")

    if camera.isOpened():
        vital, image = camera.read()
    else:
        vital = False

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        if camera.isOpened():
            vital, image = camera.read()
        else:
            vital = False

        while vital:
            vital, image = camera.read()
            cv2.imshow(cameraName, cv2.flip(image, 1))

            image.flags.writeable = False

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)#

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_styles.get_default_pose_landmarks_style()
            )
            
            if cv2.waitKey(20) == 27:
                print(f"{cameraName}_off : Pose")
                break

    camera.release()

if __name__ == "__main__":
    pose_0 = multiCamera("pose_0_Camera", 0); pose_0.start()
    hand_1 = multiCamera("hand_1_Camera", 1); hand_1.start()
    hand_2 = multiCamera("hand_2_Camera", 2); hand_2.start()
    
    #body_pose("test",0)
