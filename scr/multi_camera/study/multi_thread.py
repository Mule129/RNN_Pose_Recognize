import cv2
import threading
import mediapipe as mp

class camThread(threading.Thread):#threading.thread 클래스 상속
    def __init__(self, cameraName, id):
        threading.Thread.__init__(self)#threading 인스턴스 초기화(자동호출이 안됨)
        self.cameraName = cameraName
        self.id = id
    def run(self):
        print ("Starting " + self.cameraName)
        camPreview(self.cameraName, self.id)

def camPreview(cameraName, id):
    pose = mp.solutions.pose
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
        cv2.imshow(cameraName, cv2.flip(image, 1))

        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            pose.POSE_CONNECTIONS,
            landmark_drawing_spec = styles.get_default_pose_landmarks_style()
        )
        cv2.imshow(cameraName, cv2.flip(image, 1))
    cv2.destroyWindow(cameraName)

# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
#thread2.start()