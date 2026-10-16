"""24-joint layout shared by source and target domains."""

JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

PELVIS = 0
NECK = 12
FOOT_JOINTS = (7, 8, 10, 11)
ARM_JOINTS = tuple(range(16, 24))
LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
HEAD_JOINTS = (12, 15)
TORSO_JOINTS = (0, 3, 6, 9, 13, 14)
