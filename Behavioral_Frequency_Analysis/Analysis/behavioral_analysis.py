
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

#Custom imports
from configuration.settings import BASELINE_WINDOW_SEC, mapping, contributor_mapping
from configuration.directory_config import output_analysis_root
from utility.utils import ensure_dir, seconds_to_timestamp, sanitize_filename
from event_generation import generate_event_evidence


# Main Analysis Function
def behavioural_frequency_analysis(
    input_csv,
    output_subfolder,
    participant_id,
    video_path,
    take_metadata,
    video_metadata
):

    df = pd.read_csv(input_csv)

    print(f"📊 Processing: {participant_id}")

    time_minutes = df['Seconds'] / 60.0

    # Synthezing Expression Groups based on FACS Action Units

    df['AU12_AU6_DuchenneSmile'] = ( df[['MouthSmileLeft', 'MouthSmileRight']].mean(axis=1) + df[['CheekSquintLeft', 'CheekSquintRight']].mean(axis=1) ) / 2

    df['AU4_AU15_AU24_Concentration'] = ( ( df['BrowDownLeft'] + df['BrowDownRight'] ) / 2 + ( df['MouthFrownLeft'] + df['MouthFrownRight'] ) / 2 + ( df['MouthPressLeft'] + df['MouthPressRight'] ) / 2 + ( df['EyeSquintLeft'] + df['EyeSquintRight'] ) / 2 ) / 4
   
    df['AU1_AU2_BrowRaise'] = df[ ['BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight'] ].mean(axis=1)

    df['Asymmetric_BrowLift'] = ( df['BrowOuterUpLeft'] - df['BrowOuterUpRight'] ).abs()

    df['AU5_AU26_StartleResponse'] = df[ ['EyeWideLeft', 'EyeWideRight', 'JawOpen'] ].mean(axis=1)

    df['AU14_AU24_Tension'] = df[ ['MouthDimpleLeft', 'MouthDimpleRight', 'MouthClose'] ].mean(axis=1)

    df['AU15_LipCornerDepressor'] = df[ ['MouthFrownLeft', 'MouthFrownRight'] ].mean(axis=1)

    df['AU9_Aversion'] = ( ( df['NoseSneerLeft'] + df['NoseSneerRight'] ) / 2 + ( df['EyeSquintLeft'] + df['EyeSquintRight'] ) / 2 + ( df['BrowDownLeft'] + df['BrowDownRight'] ) / 2 ) / 3


    # baselines threshold calculations are performed exclusively within the first 30 seconds of the recording to ensure a robust representation of the participant's neutral state, minimizing the influence of transient expressions or artifacts that may occur later in the session.
    baseline_df = df[df['Seconds'] <= BASELINE_WINDOW_SEC].copy()

    print(
        f"🧪 Baseline window: "
        
        f"{BASELINE_WINDOW_SEC}s "
        f"({len(baseline_df)} frames)"
    )

    if len(baseline_df) < 10:
        print(f"⚠️ Insufficient baseline frames for {participant_id}. Falling back to full recording.")
        baseline_df = df.copy()


    summary_data = []
    event_registry = []
    smoothed_trends_dict = {}

    # Output Folders    
    reports_dir = os.path.join(output_subfolder, "reports")
    ensure_dir(reports_dir)

    plots_dir = os.path.join(output_subfolder, "plots")
    ensure_dir(plots_dir)

    # Analysis per expression group
    for tracking_key, display_name in mapping.items():

        mean_raw = baseline_df[tracking_key].mean()
        std_raw = baseline_df[tracking_key].std()

        threshold_raw = mean_raw + (2.5 * std_raw)

        peaks_raw, _ = find_peaks( df[tracking_key], height=threshold_raw, distance=30 )
        bout_count_raw = len(peaks_raw)

        # Smoothing and refined peak detection
        smoothed_series = savgol_filter( df[tracking_key], window_length=61, polyorder=2 ).clip(0.0, 1.0)
        smoothed_trends_dict[display_name] = smoothed_series # Saved for stackplot later

        baseline_mask = df['Seconds'] <= BASELINE_WINDOW_SEC
        baseline_smoothed_series = smoothed_series[baseline_mask]

        mean_smooth = baseline_smoothed_series.mean()
        std_smooth = baseline_smoothed_series.std()

        threshold_smooth = mean_smooth + (2.5 * std_smooth)

        peaks_smooth, _ = find_peaks( smoothed_series, height=threshold_smooth, distance=(60 * 2.0) )
        bout_count_smooth = len(peaks_smooth)
        print(
            f"   📈 {display_name}: "
            f"{bout_count_smooth} filtered events "
            f"(threshold={threshold_smooth:.3f})"
        )


        # SUMMARY

        summary_data.append({
            'Expression Group': display_name,
            'Raw_Mean_Baseline': mean_raw,
            'Raw_Max_Observed': df[tracking_key].max(),
            'Raw_Bout_Count_Events': bout_count_raw,
            'Smoothed_Bout_Count_Events': bout_count_smooth,
            'False_Positives_Filtered_Out': (
                bout_count_raw - bout_count_smooth
            )
        })

        # --- Visualization  ---

        # Compute 15% head-room spacing cushion dynamically
        max_raw_val = df[tracking_key].max()
        ylim_raw_ceiling = max(max_raw_val * 1.15, 0.30)
        
        max_smooth_val = smoothed_series.max()
        ylim_smooth_ceiling = max(max_smooth_val * 1.15, 0.30)

        # PLOT 1: Raw Data Timeline (With External Floating Legend)
        plt.figure(figsize=(12, 4.8))
        plt.plot(time_minutes, df[tracking_key], color='#d62728', alpha=0.7, linewidth=1, label='Raw Tracked Intensity')
        plt.axhline(y=threshold_raw, color='black', linestyle='--', alpha=0.8, label=f'Raw Active Threshold ({threshold_raw:.3f})')
        if bout_count_raw > 0:
            plt.plot(time_minutes.iloc[peaks_raw], df[tracking_key].iloc[peaks_raw], 'x', color='black', markersize=7, markeredgewidth=2, label='Raw Detected Bout Event')
            
        plt.title(f'RAW Timeline: {display_name} ({participant_id})', fontsize=12, fontweight='bold', pad=25)
        plt.xlabel('Time (Minutes)', fontweight='bold')
        plt.ylabel('Intensity (0.0 to 1.0)', fontweight='bold')
        plt.xlim(0, time_minutes.max())
        plt.ylim(-0.02, ylim_raw_ceiling)
        plt.xticks(np.arange(0, time_minutes.max() + 1, 1.0))
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Floating overhead multi-column legend (No text collisions)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), mode="expand", borderaxespad=0, ncol=3, fontsize=9)
        plt.tight_layout()
        
        
        # Dynamic filename cleaning
        safe_filename = display_name.replace("/", "_")
        plt.savefig(os.path.join(plots_dir, f"{safe_filename}_raw_plot.png"), dpi=200) 
        plt.close()
        
        # PLOT 2: Smoothed Trend Timeline
        plt.figure(figsize=(12, 4.8))
        plt.plot(time_minutes, smoothed_series, color='#1f77b4', linewidth=2, label='Smoothed Trend (Savitzky-Golay Filter)')
        plt.axhline(y=threshold_smooth, color='black', linestyle='--', alpha=0.8, label=f'Smoothed Active Threshold ({threshold_smooth:.3f})')
        if bout_count_smooth > 0:
            plt.plot(time_minutes.iloc[peaks_smooth], smoothed_series[peaks_smooth], 'o', color='#ff7f0e', markersize=6, label='Filtered Behavioral Bout')
            
        plt.title(f'SMOOTHED Trend Timeline: {display_name} ({participant_id})', fontsize=12, fontweight='bold', pad=25)
        plt.xlabel('Time (Minutes)', fontweight='bold')
        plt.ylabel('Intensity (0.0 to 1.0)', fontweight='bold')
        plt.xlim(0, time_minutes.max())
        plt.ylim(-0.02, ylim_smooth_ceiling)
        plt.xticks(np.arange(0, time_minutes.max() + 1, 1.0))
        plt.grid(True, linestyle=':', alpha=0.5)
        
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), mode="expand", borderaxespad=0, ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{safe_filename}_smoothed_plot.png"), dpi=200) 
        plt.close()

        fps = video_metadata.get("FrameRate", 60)
        recording_identifier = take_metadata.get( "identifier", "Unknown" )
        recording_date = take_metadata.get( "date", "Unknown" )

        for idx, peak_idx in enumerate(peaks_smooth):

            timestamp = float(df.iloc[peak_idx]["Seconds"])
            frame_index = int(round(timestamp * fps))
            contributors = []

            for contributor in contributor_mapping[tracking_key]:

                contributors.append(( contributor, float(df.iloc[peak_idx][contributor]) ))

            contributors = sorted( contributors, key=lambda x: x[1], reverse=True )

            safe_expression = sanitize_filename(display_name)

            event_registry.append({
                "participant": participant_id,
                "expression": display_name,
                "event_id": f"{safe_expression}_{idx + 1:03d}",
                "timestamp": timestamp,
                "frame_index": frame_index,
                "fps": fps,
                "peak_intensity": float(smoothed_series[peak_idx]),
                "threshold": float(threshold_smooth),
                "contributors": contributors,
                "recording_identifier": recording_identifier,
                "recording_date": recording_date
            })

    # Expression Frequency Summary

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv( os.path.join(reports_dir, f"{participant_id}_frequency_analysis.csv" ), index=False)

    # Save a copy to another folder:
    global_csv_dir = os.path.join(output_analysis_root, "all_results", "csv")
    os.makedirs(global_csv_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(global_csv_dir, f"{participant_id}_frequency_analysis.csv"), index=False)

    event_rows = []

    for event in event_registry:

        contributor_string = ", ".join([ f"{name} ({value:.3f})" for name, value in event["contributors"] ])

        event_rows.append({
            "Event ID": event["event_id"],
            "Expression": event["expression"],
            "Timestamp": seconds_to_timestamp(event["timestamp"]),
            "Frame Index": event["frame_index"],
            "Peak Intensity": event["peak_intensity"],
            "Threshold": event["threshold"],
            "Contributors": contributor_string
        })

    df_events = pd.DataFrame(event_rows)
    df_events.to_csv( os.path.join( reports_dir, f"{participant_id}_event_evidence.csv" ), index=False )

    # Plot 3: Combined Stackplot of All Smoothed Expression Trends
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'Master Behavioral Profile: {participant_id}', fontsize=16, fontweight='bold', y=0.99)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (display_name, data_series) in enumerate(smoothed_trends_dict.items()):
        ax = axes[i]
        ax.plot(time_minutes, data_series, color=colors[i], linewidth=2)
        
        # Clean vertical wrapping formatting on left y-axis margins
        formatted_label = display_name.replace("/", "\n")
        ax.set_ylabel(formatted_label, rotation=0, labelpad=55, va='center', fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.tick_params(axis='y', labelsize=8)
        
        # Overlay subtle dashboard baseline indicators
        baseline_mask = df['Seconds'] <= BASELINE_WINDOW_SEC

        baseline_data = data_series[baseline_mask]

        thresh_dash = (
            baseline_data.mean()
            + (2.5 * baseline_data.std())
        )
        ax.axhline(y=thresh_dash, color='red', linestyle=':', alpha=0.4)

    # Combined plot
    plt.xlabel('Time (Minutes)', fontsize=12, fontweight='bold')
    plt.xlim(0, time_minutes.max())
    plt.xticks(np.arange(0, time_minutes.max() + 1, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{participant_id}_Combined_Plot.png"), dpi=200)

    # Save a duplicate to another folder:
    global_plots_dir = os.path.join(output_analysis_root, "all_results", "plots")
    os.makedirs(global_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(global_plots_dir, f"{participant_id}_Combined_Plot.png"), dpi=200)
    plt.close()

    # Overlay Generation
    generate_event_evidence(event_registry, video_path, output_subfolder)

    print(f"\n Total detected events: {len(event_registry)}")

    print(f"Finished: {participant_id}")
