from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

#custom imports
from project_io.io_utils import extract_frame
from utility.utils import seconds_to_timestamp
from configuration.settings import EXPORT_FRAME_WIDTH, EXPORT_FRAME_HEIGHT, BEFORE_OFFSET_SEC, AFTER_OFFSET_SEC, TOP_PROMINENT_EVENTS, TEMPORAL_WINDOW_SEC, TEMPORAL_OUTPUT_FPS, BASELINE_WINDOW_SEC, RESUME_FROM, face_detector, FACIAL_REGIONS, FACIAL_CONTOURS, BLENDSHAPE_REGION_MAP
 


# Facial Heatmap Visualization for Expression Contributors
def expression_heatmap(frame_np, contributors):
    # Create a single-channel grayscale mask for the heat accumulation
    h, w, c = frame_np.shape
    heat_mask = np.zeros((h, w), dtype=np.float32)

    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    frame_bgr, faces = face_detector.findFaceMesh(frame_bgr, draw=False)

    if not faces:
        empty_overlay = np.zeros_like(frame_np)
        return frame_np, empty_overlay

    face = faces[0]

    # Create a blank black frame for drawing the wireframe canvas exclusively
    wireframe_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    wireframe_color = (220, 220, 220) 
    for p1_idx, p2_idx in FACIAL_CONTOURS:
        if p1_idx < len(face) and p2_idx < len(face):
            pt1 = tuple(face[p1_idx])
            pt2 = tuple(face[p2_idx])
            cv2.line(wireframe_canvas, pt1, pt2, wireframe_color, 1, cv2.LINE_AA)
    
    # Blend the wireframe into the base frame with exactly 25% opacity
    frame_bgr_raw = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    base_frame_with_wireframe = cv2.addWeighted(frame_bgr_raw, 0.75, wireframe_canvas, 0.25, 0)

    # 1. Accumulate weights onto the grayscale mask
    for blendshape_name, intensity in contributors:
        if blendshape_name not in BLENDSHAPE_REGION_MAP:
            continue

        target_regions = BLENDSHAPE_REGION_MAP[blendshape_name]
        for region_name in target_regions:
            landmark_indices = FACIAL_REGIONS[region_name]
            points = []

            for idx in landmark_indices:
                if idx >= len(face):
                    continue
                x, y = face[idx]
                points.append((x, y))

            if len(points) < 3:
                continue

            points = np.array(points, dtype=np.int32)
            
            poly_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(poly_mask, [points], 1.0)
            heat_mask += poly_mask * float(intensity)

    # 2. Advanced Spatial Blur Diffusion and Dynamic Normalization
    # Apply a wider filter to cleanly distribute mechanical skin deformation
    blurred_mask = cv2.GaussianBlur(heat_mask, (91, 91), 0)

    max_val = np.max(blurred_mask)
    if max_val > 0.001:
        # Normalize across the full 8-bit dynamic spectrum to guarantee visual contrast
        blurred_mask_uint8 = ((blurred_mask / max_val) * 255).astype(np.uint8)
    else:
        blurred_mask_uint8 = np.zeros((h, w), dtype=np.uint8)

    # High-contrast scientific JET profile: 0 maps to Blue, 255 maps to Red
    heatmap_color = cv2.applyColorMap(blurred_mask_uint8, cv2.COLORMAP_JET)

    # 3. Continuous Alpha-Blended Overlay Mapping
    # Convert mask to a proportional scale factor (0.0 to 1.0)
    alpha_mask = blurred_mask_uint8.astype(np.float32) / 255.0
    alpha_3d = np.expand_dims(alpha_mask, axis=2)
    
    # Restrict maximum heatmap opacity to 70% so skin textures and wireframes remain visible
    MAX_HEATMAP_OPACITY = 0.58
    blend_alpha = alpha_3d * MAX_HEATMAP_OPACITY
    
    # Execute pixel-wise continuous linear blending
    final_frame_bgr = (blend_alpha * heatmap_color.astype(np.float32)) + ((1.0 - blend_alpha) * base_frame_with_wireframe.astype(np.float32))
    final_frame_bgr = np.clip(final_frame_bgr, 0, 255).astype(np.uint8)

    # Re-wrap color channels for downstream PIL/image pipelines
    final_frame_rgb = cv2.cvtColor(final_frame_bgr, cv2.COLOR_BGR2RGB)
    final_heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return final_frame_rgb, final_heatmap_rgb

# Temporal Expression Trail Video
def create_temporal_expression_video(cap, event_data, output_video_path, video_duration):

    start_time = max(0, event_data["timestamp"] - TEMPORAL_WINDOW_SEC / 2)
    end_time = min(video_duration, event_data["timestamp"] + TEMPORAL_WINDOW_SEC / 2)
    frame_times = np.arange(start_time, end_time, 1.0 / TEMPORAL_OUTPUT_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, TEMPORAL_OUTPUT_FPS, (EXPORT_FRAME_WIDTH, EXPORT_FRAME_HEIGHT))

    peak_time = event_data["timestamp"]

    for timestamp in frame_times:
        frame_np = extract_frame(cap, timestamp, video_duration)
        if frame_np is None:
            continue

        # Temporal weighting based on proximity to peak frame
        distance_from_peak = abs(timestamp - peak_time)
        temporal_strength = max(0.0, 1.0 - (distance_from_peak / (TEMPORAL_WINDOW_SEC / 2)))

        weighted_contributors = []
        for name, value in event_data["contributors"][:5]:
            weighted_value = value * temporal_strength
            weighted_contributors.append((name, weighted_value))

        # Generate the dynamic frame blend
        final_frame, _ = expression_heatmap(frame_np.copy(), weighted_contributors)

        # Timestamp text layout string
        display_time = seconds_to_timestamp(timestamp)
        cv2.putText(
            final_frame, f"Time: {display_time}", (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Peak timestamp highlight overlay flag
        if abs(timestamp - peak_time) < 0.05:
            cv2.putText(
                final_frame, "PEAK EVENT", (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
            )

        final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(final_frame_bgr)

    video_writer.release()


# Event overlay creation
def create_event_overlay(cap, event_data, output_path, video_duration):

    before_time = max(
        0,
        event_data["timestamp"] - BEFORE_OFFSET_SEC
    )

    peak_time = event_data["timestamp"]

    after_time = min(
        video_duration,
        event_data["timestamp"] + AFTER_OFFSET_SEC
    )

    before_frame = extract_frame(cap, before_time, video_duration)
    peak_frame = extract_frame(cap, peak_time, video_duration)
    after_frame = extract_frame(cap, after_time, video_duration)

    if (
        before_frame is None
        or peak_frame is None
        or after_frame is None
    ):
        print(f"Failed extraction for {event_data['event_id']} (Frame out of bounds or corrupt). Skipping panel generation.")
        return
    
    heatmap_peak, _ = expression_heatmap(
        peak_frame.copy(),
        event_data["contributors"][:5]
    )
    
    header_height = 300
    footer_height = 320

    canvas_width = EXPORT_FRAME_WIDTH * 4

    canvas_height = (
        EXPORT_FRAME_HEIGHT
        + header_height
        + footer_height
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        color=(255, 255, 255)
    )

    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_large = ImageFont.truetype("arial.ttf", 34)
    font_medium = ImageFont.truetype("arial.ttf", 26)

    # Paste frames
    y_offset = header_height

    before_frame = Image.fromarray(before_frame)
    peak_frame = Image.fromarray(peak_frame)
    after_frame = Image.fromarray(after_frame)
    heatmap_peak = Image.fromarray(heatmap_peak)

    canvas.paste(before_frame, (0, y_offset))
    canvas.paste(peak_frame, (EXPORT_FRAME_WIDTH, y_offset))
    canvas.paste(after_frame, (EXPORT_FRAME_WIDTH * 2, y_offset))
    canvas.paste(heatmap_peak, (EXPORT_FRAME_WIDTH * 3, y_offset))

    # Textual information
    header_lines = [
        f"Participant: {event_data['participant']}",
        f"Expression: {event_data['expression']}",
        f"Event ID: {event_data['event_id']}",
        f"Peak Time: {seconds_to_timestamp(event_data['timestamp'])}",
        f"Peak Intensity: {event_data['peak_intensity']:.3f}",
        f"Threshold: {event_data['threshold']:.3f}",
        #f"Frame Index: {event_data['frame_index']}",
        #f"FPS: {event_data['fps']}",
        #f"Recording ID: {event_data['recording_identifier']}",
        #f"Recording Date: {event_data['recording_date']}"
    ]

    y_text = 15

    for line in header_lines:
        draw.text((20, y_text), line, fill="black", font=font_medium)
        y_text += 36

    # Labels

    label_y = header_height - 80

    labels = [
        ("BEFORE (-0.5s)", EXPORT_FRAME_WIDTH // 2),
        ("PEAK", EXPORT_FRAME_WIDTH + EXPORT_FRAME_WIDTH // 2),
        ("AFTER (+0.5s)", EXPORT_FRAME_WIDTH * 2 + EXPORT_FRAME_WIDTH // 2),
        ("HEATMAP", EXPORT_FRAME_WIDTH * 3 + EXPORT_FRAME_WIDTH // 2)
    ]

    for text, center_x in labels:

        text_bbox = draw.textbbox((0, 0), text, font=font_large)

        text_width = text_bbox[2] - text_bbox[0]

        draw.text(
            (center_x - text_width // 2, label_y),
            text,
            fill="black",
            font=font_large
        )
    # Contributors panel 
    contributor_y = (header_height + EXPORT_FRAME_HEIGHT + 20 )
    draw.text((20, contributor_y),"Dominant Blendshape Contributors", fill="black", font=font_large)
    contributor_y += 45

    draw.line( [(20, contributor_y), (700, contributor_y)], fill=(180, 180, 180), width=2 )

    contributor_y += 20

    for idx, (name, value) in enumerate(event_data["contributors"][:5]):

        rank_text = f"{idx + 1}."
        contributor_name = name
        contributor_value = f"{value:.3f}"

        # Column positions
        rank_x = 40
        name_x = 90

        draw.text((rank_x, contributor_y),
                   rank_text, fill="black",
                   font=font_medium)
        
        draw.text((name_x, contributor_y),
                   contributor_name, 
                   fill="black", 
                   font=font_medium)

        value_bbox = draw.textbbox((0, 0), contributor_value, font=font_medium)
        value_width = value_bbox[2] - value_bbox[0]

        draw.text(
            (650 - value_width, contributor_y),
            contributor_value,
            fill="black",
            font=font_medium
        )

        contributor_y += 48

    canvas.save(output_path)
