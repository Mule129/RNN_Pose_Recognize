from turtle import pos
import cv2
import mediapipe as mp

def body_pose(cameraName, id):
    mp_pose = mp.solutions.pose
    styles = mp.solutions.drawing_styles
    drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(id)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

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
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = styles.get_default_pose_landmarks_style()
            )
            cv2.imshow(cameraName, cv2.flip(image, 1))

            if cv2.waitKey(20) == 27:
                break

    camera.release()

body_pose("test",0)