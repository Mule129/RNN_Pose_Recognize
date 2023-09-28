import cv2
from numpy import ndarray
from typing import NamedTuple

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

    def hand_preprocess(self, image: ndarray) -> NamedTuple:
        hands_result = self.hands.process(image)
        for hand_landmarks in hands_result.multi_hand_landmarks:
            draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                get_default_hand_landmarks_style(),
                get_default_hand_connections_style()
            )
        return hands_result

    def draw_hand_randmark(self, hand_process: NamedTuple):


if __name__ is "__main__":
    cv2.VideoCapture(0)
