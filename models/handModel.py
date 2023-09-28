import mediapipe.python.solutions.pose
from pydantic import BaseModel
from mediapipe.python.solutions.hands import Hands
from typing import NamedTuple


class HandModel(BaseModel, Hands):
    handModel: NamedTuple | None = None
    