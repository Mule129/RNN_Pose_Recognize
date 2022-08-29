import os
import mediapipe as mp

mp_pose = mp.solutions.pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #pose.process("test")
    print(pose)

test = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
test.process("test")
print(test)