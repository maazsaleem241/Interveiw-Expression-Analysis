import cv2
import os
import shutil
import matplotlib.pyplot as plt

#custom improt

from configuration.settings import TOP_PROMINENT_EVENTS, output_analysis_root
from utility.utils import ensure_dir, sanitize_filename
from project_io.io_utils import create_event_overlay, create_temporal_expression_video



# Evidence reporting 
def generate_event_evidence(event_registry, video_path, participant_output_subfolder):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 60

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / fps

    evidence_root = os.path.join(participant_output_subfolder, "evidence_frames")
    ensure_dir(evidence_root)

    temporal_video_root = os.path.join(participant_output_subfolder, "temporal_expression_videos" )
    ensure_dir(temporal_video_root)

    # Compiled prominent outputs (top 3 per expression) copied to a global folder for easy access
    prominent_root = os.path.join(output_analysis_root, "all_results", "prominent_evidence")
    ensure_dir(prominent_root)

    grouped_events = {}

    for event in event_registry:
        expression = sanitize_filename(event["expression"])
        if expression not in grouped_events:
            grouped_events[expression] = []
        grouped_events[expression].append(event)

    for expression, events in grouped_events.items():

        # Setup specific output subdirectories for both images and videos
        expression_img_folder = os.path.join(evidence_root, expression)
        ensure_dir(expression_img_folder)
        
        expression_vid_folder = os.path.join(temporal_video_root, expression)
        ensure_dir(expression_vid_folder)

        sorted_events = sorted(events, key=lambda x: x["peak_intensity"], reverse=True)

        for idx, event in enumerate(sorted_events):
            filename = f"{event['event_id']}.png"
            output_path = os.path.join(expression_img_folder, filename)
            
            # Generate sorted temporal expression trail video asset
            video_filename = f"{event['event_id']}.mp4"
            video_output_path = os.path.join(expression_vid_folder, video_filename)

            #Resume Check: If the output image and video files are already on disk, skip processing
            if os.path.exists(output_path) and os.path.exists(video_output_path):
                print(f"     Skipping event {event['event_id']} (Files already processed).")
                continue
            print(
                f"      Generating panel for "
                f"{event['expression']} | "
                f"{event['event_id']} | "
                f"Peak={event['peak_intensity']:.3f}"
            )

            # Generate static 4-panel analysis plot
            create_event_overlay(cap, event, output_path, video_duration)
            create_temporal_expression_video(cap, event, video_output_path, video_duration)

            # Only continue if the overlay image was successfully created
            if not os.path.exists(output_path):
                print(f"     ⚠️ Overlay generation failed for {event['event_id']}. Skipping copy.")
                continue

            # Copy top validation landmarks to global metrics pool folder
            if idx < TOP_PROMINENT_EVENTS:
                global_expression_dir = os.path.join(prominent_root, expression)
                ensure_dir(global_expression_dir)
                shutil.copy2(output_path, os.path.join(global_expression_dir, filename))

    cap.release()