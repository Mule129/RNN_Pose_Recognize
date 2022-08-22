import multiprocessing
import cv2
import mediapipe as mp

#mediapipe AttributeError 해결하기
#line 40, 79 멀티프로세싱으로 생기는 오류는 x, 지역변수나 접근권한과 관련있는듯

class multiCamera(multiprocessing.Process):
    def __init__(self, cameraName, id):
        multiprocessing.Process.__init__(self)
        self.cameraName = cameraName
        self.id = id

    def run(self):
        print(f"{self.cameraName} starting, usb port id : {self.id}")
        if id == 0:
            body_pose(self.cameraName, self.id)
        else:
            hand_pose(self.cameraName, self.id)
        pass

def hand_pose(cameraName, id):#데코레이터 참고하기
    hand = mp.solutions.hands
    styles = mp.solutions.drawing_styles
    drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(id)
    print(camera.isOpened())
    if camera.isOpened():
        vital, image = camera.read()
    else:
        vital = False

    while vital:
        vital, image = camera.read()

        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    hand.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style()
        )
        cv2.imshow(cameraName, cv2.flip(image, 1))

        if cv2.waitKey(20) == 27:
            break

    camera.release()

def body_pose(cameraName, id):
    pose = mp.solutions.pose
    styles = mp.solutions.drawing_styles
    drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(id)
    print(camera.isOpened())
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
        drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            pose.POSE_CONNECTIONS,
            landmark_drawing_spec = styles.get_default_pose_landmarks_style()
        )
        cv2.imshow(cameraName, cv2.flip(image, 1))

        if cv2.waitKey(20) == 27:
            break

    camera.release()

if __name__ == "__main__":
    test = multiCamera("camera_1", 0)
    test2 = multiCamera("camera_2", 1)
    #test.start()
    #test2.start()
    body_pose("test",0)
