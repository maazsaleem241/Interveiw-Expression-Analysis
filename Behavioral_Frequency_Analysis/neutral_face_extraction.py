import os
import cv2
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_WINDOW_SEC = 25

input_cleaned_folder = r"D:\Intern work\Data for research\Data_Cleaned"
video_root_directory = r"D:\Intern work\Data for research\인터뷰_260403"
output_root = r"D:\Intern work\Data for research\Neutral_Faces"
os.makedirs(output_root, exist_ok=True)

# Matchign Logic: Find video sessions corresponding to each participant CSV

def find_participant_resources(video_root_directory, participant_id):

    participant_number = participant_id.upper().replace("P", "")
    matching_sessions = []

    if not os.path.exists(video_root_directory):
        print(f" !! Video root directory missing: {video_root_directory}")
        return []

    for folder_name in os.listdir(video_root_directory):

        folder_path = os.path.join(video_root_directory, folder_name)

        if not os.path.isdir(folder_path):
            continue

        folder_parts = folder_name.split("_")

        # Handles:
        # 04_1_name
        # 04_2_name
        if len(folder_parts) >= 2 and folder_parts[1].isdigit():
            folder_prefix = f"{folder_parts[0]}_{folder_parts[1]}"
        else:
            folder_prefix = folder_parts[0]

        if folder_prefix == participant_number:

            mov_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".mov")
            ]

            if len(mov_files) == 0:
                print(f" !! No MOV file in {folder_name}")
                continue

            video_path = os.path.join(folder_path, mov_files[0])

            matching_sessions.append({
                "video_path": video_path,
                "folder_name": folder_name
            })

    return matching_sessions


# Neutral Score Computation
def compute_expression_groups(df):

    # 1. Positive Social Expression / Smile
    df['Smile'] = ( df[['MouthSmileLeft', 'MouthSmileRight']].mean(axis=1) + df[['CheekSquintLeft', 'CheekSquintRight']].mean(axis=1) ) / 2

    # 2. Concentration / Negative Emotion
    df['Concentration'] = ( (df['BrowDownLeft'] + df['BrowDownRight']) / 2 + (df['MouthFrownLeft'] + df['MouthFrownRight']) / 2 + (df['MouthPressLeft'] + df['MouthPressRight']) / 2 + (df['EyeSquintLeft'] + df['EyeSquintRight']) / 2 ) / 4
    
    # 3. Attentional Engagement / Brow Raise
    df['BrowRaise'] = df[ ['BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight'] ].mean(axis=1)

    # 4. Skepticism / Asymmetric Brow Lift 
    df['Skepticism'] = (df['BrowOuterUpLeft'] - df['BrowOuterUpRight']).abs()

    # 5. Startle Response / Shock
    df['Startle'] = df[ ['EyeWideLeft', 'EyeWideRight', 'JawOpen'] ].mean(axis=1)

    # 6. Tension / Stress
    df['Tension'] = df[ ['MouthDimpleLeft', 'MouthDimpleRight', 'MouthClose'] ].mean(axis=1)

    # 7. Sadness / Dejection / Lip Corner Depresson
    df['Dejection'] = df[ ['MouthFrownLeft', 'MouthFrownRight'] ].mean(axis=1)

    # 8. Aversion / Critical Evaluation
    df['Aversion'] = ( (df['NoseSneerLeft'] + df['NoseSneerRight']) / 2 + (df['EyeSquintLeft'] + df['EyeSquintRight']) / 2 + (df['BrowDownLeft'] + df['BrowDownRight']) / 2 ) / 3

    return df

# ============================================================
# EXTRACT MOST NEUTRAL FRAME
# ============================================================

def extract_neutral_face(csv_path, video_path, output_path):

    print(f"\n📊 Processing CSV: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)

    if 'Seconds' not in df.columns:
        print(" !! Missing Seconds column.")
        return

    # Restrict to baseline window
    baseline_df = df[df['Seconds'] <= BASELINE_WINDOW_SEC].copy()

    if len(baseline_df) < 30:
        print(" !!  Not enough baseline frames.")
        return

    # Compute expression groups
    baseline_df = compute_expression_groups(baseline_df)

    expression_columns = [
        'Smile', 'Concentration', 'BrowRaise', 'Skepticism', 
        'Startle', 'Tension', 'Dejection', 'Aversion'
    ]

    # Smooth each expression signal
    for col in expression_columns:

        smoothed = savgol_filter(
            baseline_df[col],
            window_length=61,
            polyorder=2
        )

        baseline_df[f"{col}_smooth"] = np.clip(smoothed, 0.0, 1.0)

    # Filtering bad frames

    valid_mask = np.ones(len(baseline_df), dtype=bool)

    # Remove speaking frames
    if 'JawOpen' in baseline_df.columns:
        valid_mask &= baseline_df['JawOpen'] < 0.05

    # Remove blink frames if available
    if 'EyeBlinkLeft' in baseline_df.columns:
        valid_mask &= baseline_df['EyeBlinkLeft'] < 0.4

    if 'EyeBlinkRight' in baseline_df.columns:
        valid_mask &= baseline_df['EyeBlinkRight'] < 0.4

    filtered_df = baseline_df[valid_mask].copy()

    if len(filtered_df) == 0:
        print(" !! No valid neutral candidates.")
        return



    # Neutrality score: Lower is more neutral

    filtered_df['NeutralityScore'] = (
        filtered_df['Smile_smooth'] +
        filtered_df['Concentration_smooth'] +
        filtered_df['BrowRaise_smooth'] +
        filtered_df['Skepticism_smooth'] +
        filtered_df['Startle_smooth'] +
        filtered_df['Tension_smooth'] +
        filtered_df['Dejection_smooth'] +
        filtered_df['Aversion_smooth']
    )

    # Lowest activation frame is most neutral candidate

    sorted_candidates = (
        filtered_df
        .sort_values(by='NeutralityScore', ascending=True)
    )

    top_candidates_list = []
    MIN_TIME_SEPARATION_SEC = 1.0

    for _, row in sorted_candidates.iterrows():

        candidate_time = float(row['Seconds'])

        # Check temporal distance from already selected frames
        too_close = False

        for existing_row in top_candidates_list:

            existing_time = float(existing_row['Seconds'])
            if abs(candidate_time - existing_time) < MIN_TIME_SEPARATION_SEC:
                too_close = True
                break

        # Accept only temporally distinct candidates

        if not too_close:
            top_candidates_list.append(row)

        # Stop once we have 5 good candidates
        if len(top_candidates_list) >= 5:
            break

        if not top_candidates_list:
            print(" !! Zero valid unique candidates met temporal diversification rules.")
            return

    # Convert back to DataFrame
    top_candidates = pd.DataFrame(top_candidates_list)
    
    print("\n Top Neutral Candidates:") 

    for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1): 
        timestamp = float(row['Seconds']) 
        score = float(row['NeutralityScore']) 
        
        print( f" {rank}. " f"Time={timestamp:.2f}s | " f"Score={score:.4f}" )

    #Video extraction of top candidates

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f" !! Failed opening video: {video_path}")
        return

    # Create dedicated folder per participant/session 
    participant_output_dir = os.path.splitext(output_path)[0] 
    os.makedirs(participant_output_dir, exist_ok=True)

    saved_candidates = [] 
    
    for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1): 

        timestamp = float(row['Seconds']) 
        score = float(row['NeutralityScore']) 
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000) 
        success, frame = cap.read() 

        if not success: 
            print(f" !! Failed reading candidate frame {rank}") 
            continue 
        
        candidate_filename = ( f"candidate_{rank:02d}" f"_t{timestamp:.2f}" f"_score{score:.4f}.png" ) 
        candidate_path = os.path.join( participant_output_dir, candidate_filename ) 
        
        cv2.imwrite(candidate_path, frame) 
        saved_candidates.append(candidate_path) 

        print( f"💾 Saved Candidate {rank}: " f"{candidate_filename}" ) 

    cap.release() 
    print( f"\n Extracted " f"{len(saved_candidates)} neutral candidates.")

# Execution Loop
print("🚀 Neutral Face Extraction Pipeline Starting...")

csv_files = [
    f for f in os.listdir(input_cleaned_folder)
    if f.lower().endswith(".csv")
]

if len(csv_files) == 0:
    print("⚠️ No CSV files found.")

for csv_file in csv_files:

    participant_base_id = os.path.splitext(csv_file)[0]

    csv_path = os.path.join(
        input_cleaned_folder,
        csv_file
    )

    matched_sessions = find_participant_resources(
        video_root_directory,
        participant_base_id
    )

    if not matched_sessions:
        print(f"⚠️ No sessions found for {participant_base_id}")
        continue

    for idx, session in enumerate(matched_sessions):

        # Handles:
        # P04_1
        # P04_2
        if len(matched_sessions) > 1:
            session_id = f"{participant_base_id}_{idx + 1}"
        else:
            session_id = participant_base_id

        print(f"\n🎬 Session: {session_id}")
        print(f"   Folder: {session['folder_name']}")

        output_filename = f"{session_id}.png"

        output_path = os.path.join(
            output_root,
            output_filename
        )

        extract_neutral_face(
            csv_path=csv_path,
            video_path=session['video_path'],
            output_path=output_path
        )

print("\n🎉 Neutral face extraction complete.")