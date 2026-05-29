from cvzone.FaceMeshModule import FaceMeshDetector

#Configuration file for the project

EXPORT_FRAME_WIDTH = 720
EXPORT_FRAME_HEIGHT = 540

BEFORE_OFFSET_SEC = 0.5
AFTER_OFFSET_SEC = 0.5

TOP_PROMINENT_EVENTS = 3
TEMPORAL_WINDOW_SEC = 2.5
TEMPORAL_OUTPUT_FPS = 20
BASELINE_WINDOW_SEC = 25
RESUME_FROM = "P04" #To resume from a certain ID in case of interruption

face_detector = FaceMeshDetector(
    staticMode=True,
    maxFaces=1
)


#Facial Maps
FACIAL_REGIONS = {
    "mouth_smile": [61, 291, 146, 375, 78, 308],
    "mouth_frown": [17, 84, 314, 181, 405],
    "oral_compression": [61, 291, 78, 308, 95, 324],
    "jaw_open": [152, 148, 176, 149],
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "inner_brow": [9, 55, 285],
    "left_eye": [33, 133, 160, 159, 158, 144],
    "right_eye": [362, 263, 387, 386, 385, 373],
    "nose": [1, 2, 98, 327]
}

# Standard Face Mesh connection pairs for clean structural visualization
FACIAL_CONTOURS = [
    # Lips outline
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    # Left Eye outline
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    # Right Eye outline
    (362, 387), (387, 385), (385, 263), (263, 373), (373, 380), (380, 362),
    # Eyebrows
    (70, 63), (63, 105), (105, 66), (66, 107),
    (336, 296), (296, 334), (334, 293), (293, 300),
    # Face Outline / Jawline silhouette
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323),
    (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93),
    (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]


# Blendshape to Facial Region Mapping for Contributor Analysis in Event Overlays. This is a simplified mapping based on common FACS Action Units and their typical facial regions. It can be further refined based on the specific blendshapes available in the dataset and their anatomical relevance.
BLENDSHAPE_REGION_MAP = {

    # Positive Social Expression
    "MouthSmileLeft": ["mouth_smile"],
    "MouthSmileRight": ["mouth_smile"],
    "CheekSquintLeft": ["left_eye"],
    "CheekSquintRight": ["right_eye"],

    # Concentration / Tension
    "MouthPressLeft": ["oral_compression"],
    "MouthPressRight": ["oral_compression"],

    "MouthFrownLeft": ["mouth_frown"],
    "MouthFrownRight": ["mouth_frown"],

    "MouthClose": ["oral_compression"],

    "MouthDimpleLeft": ["oral_compression"],
    "MouthDimpleRight": ["oral_compression"],

    # Jaw / Startle
    "JawOpen": ["jaw_open"],

    # Brow Activity
    "BrowOuterUpLeft": ["left_brow"],
    "BrowOuterUpRight": ["right_brow"],
    "BrowInnerUp": ["inner_brow"],

    "BrowDownLeft": ["left_brow"],
    "BrowDownRight": ["right_brow"],

    # Eye Activity
    "EyeWideLeft": ["left_eye"],
    "EyeWideRight": ["right_eye"],

    "EyeSquintLeft": ["left_eye"],
    "EyeSquintRight": ["right_eye"],

    # Nose / Aversion
    "NoseSneerLeft": ["nose"],
    "NoseSneerRight": ["nose"]
}


#Mapping:

mapping = {
        'AU12_AU6_DuchenneSmile': 'Positive Social Expression',
        'AU4_AU15_AU24_Concentration': 'Concentration',
        'AU1_AU2_BrowRaise': 'Attentional Engagement',
        'Asymmetric_BrowLift': 'Skepticism',
        'AU5_AU26_StartleResponse': 'Startle Response',
        'AU14_AU24_Tension': 'Tension/Stress',
        'AU15_LipCornerDepressor': 'Dejection',
        'AU9_Aversion': 'Aversion'
    }

contributor_mapping = {
        'AU12_AU6_DuchenneSmile': ['MouthSmileLeft', 'MouthSmileRight', 'CheekSquintLeft', 'CheekSquintRight'],
        'AU4_AU15_AU24_Concentration': ['BrowDownLeft', 'BrowDownRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthPressLeft', 'MouthPressRight', 'EyeSquintLeft', 'EyeSquintRight'],
        'AU1_AU2_BrowRaise': ['BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight'],
        'Asymmetric_BrowLift': ['BrowOuterUpLeft', 'BrowOuterUpRight'],
        'AU5_AU26_StartleResponse': ['EyeWideLeft', 'EyeWideRight', 'JawOpen'],
        'AU14_AU24_Tension': ['MouthDimpleLeft', 'MouthDimpleRight', 'MouthClose'],
        'AU15_LipCornerDepressor': ['MouthFrownLeft', 'MouthFrownRight'],
        'AU9_Aversion': ['NoseSneerLeft', 'NoseSneerRight', 'EyeSquintLeft', 'EyeSquintRight', 'BrowDownLeft', 'BrowDownRight']
    }


