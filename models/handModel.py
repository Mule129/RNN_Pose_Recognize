import mediapipe.python.solutions.pose
from pydantic import BaseModel
from mediapipe.python.solutions.hands import Hands
from typing import NamedTuple, Optional


class HandModel(BaseModel, Hands):
    handModel: list[Optional[NamedTuple]]
    