import os
import cv2
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_WINDOW_SEC = 25

# Updated directory paths pointing to your new unified structure
input_cleaned_folder = r"D:\Intern work\Data for research\data_rearranged\time_normalized_data"
video_root_directory = r"D:\Intern work\Data for research\data_rearranged\interviews"
output_root = r"D:\Intern work\Data for research\Neutral_Faces"
os.makedirs(output_root, exist_ok=True)

# Explicit target tracking mapping for your 12 core archetype participants
TARGET_PARTICIPANTS = {
    "pr_1": "송은성", "pr_3": "이경호", "pr_4": "김혜선", "pr_8": "이정수", 
    "pr_9": "이철호", "pr_10": "박연주", "pr_11": "김상욱", "pr_15": "김동준", 
    "pr_17": "박성호", "pr_21": "박경한", "pr_22": "최권섭", "pr_23": "박병희"
}

# ============================================================
# ADAPTED RESOURCE MATCHING LOGIC
# ============================================================

def find_participant_resources(video_root, participant_base_id):
    """
    Natively matches the standardized file string (e.g., 'pr_1' or 'pr_5_1')
    to the corresponding physical directory on your drive.
    """
    if not os.path.exists(video_root):
        print(f" !! Video root directory missing: {video_root}")
        return None

    # Identify base token (e.g., extract 'pr_5' from 'pr_5_1')
    parts = participant_base_id.split('_')
    if len(parts) >= 2:
        base_pr_id = f"{parts[0]}_{parts[1]}"
    else:
        return None

    # Strict check: verify if the participant belongs to the target groups
    if base_pr_id not in TARGET_PARTICIPANTS:
        return None

    # Look for the directory that matches this specific session identifier
    for folder_name in os.listdir(video_root):
        folder_path = os.path.join(video_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Natively handles both 'pr_1_송은성' and 'pr_5_1_채명순' layouts directly
        if folder_name.startswith(f"{participant_base_id}_") or folder_name == participant_base_id:
            mov_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mov")]
            
            if len(mov_files) == 0:
                print(f" !! No MOV file discovered inside folder: {folder_name}")
                return None
                
            return {
                "video_path": os.path.join(folder_path, mov_files[0]),
                "folder_name": folder_name
            }
            
    return None

# ============================================================
# EXPRESSION MATH & FILTRATION (UNALTERED PRINCIPLES)
# ============================================================

def compute_expression_groups(df):
    df['Smile'] = (df[['MouthSmileLeft', 'MouthSmileRight']].mean(axis=1) + df[['CheekSquintLeft', 'CheekSquintRight']].mean(axis=1)) / 2

    df['Concentration'] = ((df['BrowDownLeft'] + df['BrowDownRight']) / 2 + 
                           (df['MouthFrownLeft'] + df['MouthFrownRight']) / 2 + 
                           (df['MouthPressLeft'] + df['MouthPressRight']) / 2 + 
                           (df['EyeSquintLeft'] + df['EyeSquintRight']) / 2) / 4
    
    df['BrowRaise'] = df[['BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight']].mean(axis=1)
    df['Skepticism'] = (df['BrowOuterUpLeft'] - df['BrowOuterUpRight']).abs()
    df['Startle'] = df[['EyeWideLeft', 'EyeWideRight', 'JawOpen']].mean(axis=1)
    df['Tension'] = df[['MouthDimpleLeft', 'MouthDimpleRight', 'MouthClose']].mean(axis=1)
    df['Dejection'] = df[['MouthFrownLeft', 'MouthFrownRight']].mean(axis=1)
    df['Aversion'] = ((df['NoseSneerLeft'] + df['NoseSneerRight']) / 2 + 
                      (df['EyeSquintLeft'] + df['EyeSquintRight']) / 2 + 
                      (df['BrowDownLeft'] + df['BrowDownRight']) / 2) / 3
    return df

def extract_neutral_face(csv_path, video_path, output_path):
    print(f"📊 Processing CSV: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)

    if 'Seconds' not in df.columns:
        print(" !! Missing Seconds column.")
        return

    baseline_df = df[df['Seconds'] <= BASELINE_WINDOW_SEC].copy()
    if len(baseline_df) < 30:
        print(" !! Not enough baseline data points present.")
        return

    baseline_df = compute_expression_groups(baseline_df)
    expression_columns = ['Smile', 'Concentration', 'BrowRaise', 'Skepticism', 'Startle', 'Tension', 'Dejection', 'Aversion']

    for col in expression_columns:
        smoothed = savgol_filter(baseline_df[col], window_length=61, polyorder=2)
        baseline_df[f"{col}_smooth"] = np.clip(smoothed, 0.0, 1.0)

    valid_mask = np.ones(len(baseline_df), dtype=bool)
    if 'JawOpen' in baseline_df.columns:
        valid_mask &= baseline_df['JawOpen'] < 0.05
    if 'EyeBlinkLeft' in baseline_df.columns:
        valid_mask &= baseline_df['EyeBlinkLeft'] < 0.4
    if 'EyeBlinkRight' in baseline_df.columns:
        valid_mask &= baseline_df['EyeBlinkRight'] < 0.4

    filtered_df = baseline_df[valid_mask].copy()
    if len(filtered_df) == 0:
        print(" !! Zero valid candidates left post-blink/speech isolation filtering.")
        return

    filtered_df['NeutralityScore'] = (
        filtered_df['Smile_smooth'] + filtered_df['Concentration_smooth'] +
        filtered_df['BrowRaise_smooth'] + filtered_df['Skepticism_smooth'] +
        filtered_df['Startle_smooth'] + filtered_df['Tension_smooth'] +
        filtered_df['Dejection_smooth'] + filtered_df['Aversion_smooth']
    )

    sorted_candidates = filtered_df.sort_values(by='NeutralityScore', ascending=True)
    top_candidates_list = []
    MIN_TIME_SEPARATION_SEC = 1.0

    for _, row in sorted_candidates.iterrows():
        candidate_time = float(row['Seconds'])
        too_close = False
        for existing_row in top_candidates_list:
            existing_time = float(existing_row['Seconds'])
            if abs(candidate_time - existing_time) < MIN_TIME_SEPARATION_SEC:
                too_close = True
                break
        if not too_close:
            top_candidates_list.append(row)
        if len(top_candidates_list) >= 5:
            break

    if not top_candidates_list:
        print(" !! No candidates passed temporal variance rules.")
        return

    top_candidates = pd.DataFrame(top_candidates_list)
    
    print(" Top Neutral Candidates Evaluated:") 
    for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1): 
        print(f"  {rank}. Time={float(row['Seconds']):.2f}s | Score={float(row['NeutralityScore']):.4f}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" !! Cannot mount video stream resource: {video_path}")
        return

    # Saves frames cleanly into a dedicated subfolder matching the structured name
    participant_output_dir = os.path.splitext(output_path)[0] 
    os.makedirs(participant_output_dir, exist_ok=True)

    saved_count = 0
    for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1): 
        timestamp = float(row['Seconds']) 
        score = float(row['NeutralityScore']) 
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000) 
        success, frame = cap.read() 

        if not success: 
            continue 
        
        candidate_filename = f"candidate_{rank:02d}_t{timestamp:.2f}_score{score:.4f}.png"
        cv2.imwrite(os.path.join(participant_output_dir, candidate_filename), frame) 
        saved_count += 1
        print(f"  💾 Saved Candidate {rank}: {candidate_filename}") 

    cap.release() 
    print(f" ✓ Successfully completed frame extraction processing context.\n")

# ============================================================
# RUNBATCH EXECUTION LOOP
# ============================================================
if __name__ == "__main__":
    print("🚀 Balanced Framework Neutral Face Extraction Processing Initiated...")

    if not os.path.exists(input_cleaned_folder):
        print(f"❌ Error: Time-normalized source directory missing: {input_cleaned_folder}")
        exit()

    csv_files = [f for f in os.listdir(input_cleaned_folder) if f.lower().endswith(".csv")]

    for csv_file in csv_files:
        # e.g., parses 'pr_1' from 'pr_1.csv' or 'pr_5_1' from 'pr_5_1.csv'
        session_id = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(input_cleaned_folder, csv_file)

        # Match data straight against your clean targeted repository structure
        session_resource = find_participant_resources(video_root_directory, session_id)

        if session_resource is None:
            # Safely filters out middle-zone files if they remain in your folders
            continue

        print(f"🎬 Active Targeted Session ID: {session_id}")
        print(f"   Target Folder Matrix: {session_resource['folder_name']}")

        output_path = os.path.join(output_root, f"{session_id}.png")

        extract_neutral_face(
            csv_path=csv_path,
            video_path=session_resource['video_path'],
            output_path=output_path
        )

    print("🎉 All targeted neutral face extractions complete.")