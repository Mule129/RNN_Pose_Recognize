import multiprocessing
import cv2
import mediapipe as mp

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
            body_pose(self.cameraName, self.id)

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
            cv2.imshow(cameraName, cv2.flip(image, 1))

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
            cv2.imshow(cameraName, cv2.flip(image, 1))

            if cv2.waitKey(20) == 27:
                print(f"{cameraName}_off : Pose")
                break

    camera.release()

if __name__ == "__main__":
    pose_0 = multiCamera("pose_0_Camera", 0); pose_0.start()
    hand_1 = multiCamera("hand_1_Camera", 1); hand_1.start()
    hand_2 = multiCamera("hand_2_Camera", 2); hand_2.start()
    
    #body_pose("test",0)
