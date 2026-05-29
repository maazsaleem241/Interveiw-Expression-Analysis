import os

# Custom imports
from Analysis.behavioral_analysis import behavioural_frequency_analysis
from project_io.io_utils import find_participant_resources
from configuration.settings import RESUME_FROM
from configuration.directory_config import input_cleaned_folder, video_root_directory, output_analysis_root
from utility.utils import ensure_dir


# Main execution script for Behavioral Frequency Analysis Pipeline

if __name__ == "__main__":

    print("Initializing Behavioral Evidence Pipeline...")

    resume_mode = False

    all_files = os.listdir(input_cleaned_folder)

    csv_files = [
        f for f in all_files
        if f.lower().endswith('.csv')
    ]

    if not csv_files:
        print("No CSV files found.")

    else:

        print(f"Found {len(csv_files)} CSV files in clean index directory.")

        for csv_file_name in csv_files:

            # Extract participant label
            participant_base_id = os.path.splitext(csv_file_name)[0]

            # Resume logic
            if participant_base_id.startswith(RESUME_FROM):
                resume_mode = True
                print(f"▶️ Resuming from participant: {participant_base_id}")

            # Skip earlier participants
            if not resume_mode:
                print(f"⏭️ Skipping participant: {participant_base_id}")
                continue

            full_input_path = os.path.join(
                input_cleaned_folder,
                csv_file_name
            )

            # Find all matching sessions
            matched_sessions = find_participant_resources(
                video_root_directory,
                participant_base_id
            )

            if not matched_sessions:
                print(f"Skipping baseline extraction for: {participant_base_id}")
                continue

            # Iterate through all sessions
            for idx, session in enumerate(matched_sessions):

                if len(matched_sessions) > 1:
                    session_id = f"{participant_base_id}_{idx + 1}"
                else:
                    session_id = participant_base_id

                print(f"\n🎬 Matched Active Directory Sequence: {session_id}")

                print(f"   Folder Location: {session['folder_name']}")

                print(f"   Target Video   : {os.path.basename(session['video_path'])}")

                participant_output_subfolder = os.path.join(
                    output_analysis_root,
                    session_id
                )

                ensure_dir(participant_output_subfolder)

                behavioural_frequency_analysis(
                    input_csv=full_input_path,
                    output_subfolder=participant_output_subfolder,
                    participant_id=session_id,
                    video_path=session['video_path'],
                    take_metadata=session['take_metadata'],
                    video_metadata=session['video_metadata']
                )

        print("Full pipeline completed.")