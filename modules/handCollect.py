import cv2
from typing import NamedTuple, Optional
from numpy import ndarray
from models import handModel

from mediapipe.python.solutions.hands import (
    Hands,
    HandLandmark,
    HAND_CONNECTIONS
)
from mediapipe.python.solutions.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_hand_connections_style
)
from mediapipe.python.solutions.drawing_utils import draw_landmarks

from models.handModel import HandModel


class HandCollect:
    def __init__(self):
        self.hands = Hands(
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )

    def save_array(self, array: ndarray):
        pass

    def hand_preprocess(self, hand_image: ndarray) -> tuple[ndarray, Optional[NamedTuple]]:
        hands_result = self.hands.process(hand_image)
        try:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                draw_landmarks(
                    hand_image,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    get_default_hand_landmarks_style(),
                    get_default_hand_connections_style()
                )
            return hand_image, hands_result
        except TypeError:
            return hand_image, None

    @staticmethod
    def collect_hand(self, hand_process: list[NamedTuple]) -> HandModel:
        model = HandModel(handModel=hand_process)
        return model


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    hand = HandCollect()
    WINDOW_NAME = "HandDataCollector"

    vital_pose = 0
    processing_image = None
    while True:
        vital, image = camera.read()

        if vital != 1:
            raise Exception("Camera is not open")

        wait_key = cv2.waitKey(10)
        vital_pose = wait_key if wait_key != -1 else vital_pose

        if vital_pose == 103:  # go(g, ㅎ) - 78, 103
            processing_image = hand.hand_preprocess(image)
            cv2.imshow(WINDOW_NAME, image)
            hand.save_array(image)

        elif vital_pose == 27:  # esc key
            break
        else:
            cv2.imshow(WINDOW_NAME, image)


