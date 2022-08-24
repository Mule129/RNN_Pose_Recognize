import cv2
import threading

class camThread(threading.Thread):#threading.thread 클래스 상속
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)#threading 인스턴스 초기화(자동호출이 안됨)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print ("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 1)
thread1.start()
thread2.start()