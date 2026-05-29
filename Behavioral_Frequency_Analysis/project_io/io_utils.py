import json
import os
import cv2
import numpy as np
from configuration.settings import EXPORT_FRAME_WIDTH, EXPORT_FRAME_HEIGHT, BEFORE_OFFSET_SEC, AFTER_OFFSET_SEC, TOP_PROMINENT_EVENTS, TEMPORAL_WINDOW_SEC, TEMPORAL_OUTPUT_FPS, BASELINE_WINDOW_SEC, RESUME_FROM, face_detector, FACIAL_REGIONS, FACIAL_CONTOURS, BLENDSHAPE_REGION_MAP
 

# io utilities for the project:

# Loading Json Metadata
def load_take_metadata(input_folder):
    metadata_path = os.path.join(input_folder, "take.json")

    if not os.path.exists(metadata_path):
        print("⚠️ take.json not found")
        return {}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata

def load_video_metadata(input_folder):
    metadata_path = os.path.join(input_folder, "video_metadata.json")

    if not os.path.exists(metadata_path):
        print("⚠️ video_metadata.json not found")
        return {}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata

# Function to ensure synchronization between mov and csv files for each participant, and to return the video path for the evidence generation step
def find_participant_resources(video_root_directory, participant_id):

    # P01 -> 01
    participant_number = participant_id.upper().replace("P", "")
    matching_sessions = []

    if not os.path.exists(video_root_directory):
        print(f"⚠️ Video root directory does not exist: {video_root_directory}")
        return []

    for folder_name in os.listdir(video_root_directory):

        folder_path = os.path.join(video_root_directory, folder_name)

        if not os.path.isdir(folder_path):
            continue

        folder_parts = folder_name.split("_")

        # Handle cases like 04_1_name
        if len(folder_parts) >= 2 and folder_parts[1].isdigit():
            folder_prefix = f"{folder_parts[0]}_{folder_parts[1]}"
        else:
            folder_prefix = folder_parts[0]

        # Match folder starting with participant number
        if folder_prefix == participant_number:
            
            mov_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".mov")
            ]

            if len(mov_files) == 0:
                print(f"⚠️ No MOV file inside {folder_name}")
                continue

            if len(mov_files) > 1:
                print(f"⚠️ Multiple MOV files found in {folder_name}. Using the first one.")

            video_path = os.path.join(folder_path, mov_files[0])
            take_metadata = load_take_metadata(folder_path)
            video_metadata = load_video_metadata(folder_path)

            matching_sessions.append({
                "video_path": video_path,
                "take_metadata": take_metadata,
                "video_metadata": video_metadata,
                "folder_name": folder_name
            })

    if not matching_sessions:
        print(f"⚠️ No matching directories found for participant {participant_id}")
        return []
    return matching_sessions

#Frame extraction
def extract_frame(cap, timestamp_sec, video_duration):

    timestamp_sec = max(0, min(timestamp_sec, video_duration))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    success, frame = cap.read()

    if not success:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = frame.shape[:2]

    scale = min( EXPORT_FRAME_WIDTH / w, EXPORT_FRAME_HEIGHT / h )

    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = cv2.resize(frame, (new_w, new_h))
    canvas_frame = np.ones( (EXPORT_FRAME_HEIGHT, EXPORT_FRAME_WIDTH, 3), dtype=np.uint8 ) * 255
    x_offset = (EXPORT_FRAME_WIDTH - new_w) // 2
    y_offset = (EXPORT_FRAME_HEIGHT - new_h) // 2
    canvas_frame[ y_offset:y_offset + new_h, x_offset:x_offset + new_w ] = frame

    frame = canvas_frame

    return frame