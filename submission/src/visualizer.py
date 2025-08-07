import cv2
import mediapipe as mp
import os
import json
import cv2 # Added cv2 import
from feedback_analyzer import calculate_angle, analyze_pose
from mediapipe.framework.formats import landmark_pb2 # Import NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def visualize_feedback(video_name, frames_dir, poses_dir, output_dir, current_video_feedback_data, fps):
    """
    Overlays detected landmarks and feedback text on frame images and saves them as feedback videos for a single video.

    Args:
        video_name (str): The name of the video being processed.
        frames_dir (str): Directory containing the original video frames.
        poses_dir (str): Directory containing the JSON files with pose landmark data.
        output_dir (str): Directory to save the output feedback videos.
        current_video_feedback_data (dict): The feedback data for the current video.
        fps (int): Frames per second of the video.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_frames_path = os.path.join(frames_dir, video_name)
    actual_poses_path = os.path.join(poses_dir, video_name)

    if os.path.isdir(video_frames_path) and os.path.isdir(actual_poses_path):
            output_video_path = os.path.join(output_dir, f"{video_name}_feedback.mp4")

            
            # Get frame dimensions from the first frame to initialize VideoWriter
            first_frame_path = sorted([os.path.join(video_frames_path, f) for f in os.listdir(video_frames_path) if f.endswith('.jpg') or f.endswith('.png')])
            if not first_frame_path:
                print(f"Warning: No frames found for video {video_name}. Skipping.")
                return
            
            try:
                first_frame = cv2.imread(first_frame_path[0])
                if first_frame is None:
                    print(f"Warning: Could not read the first frame {first_frame_path[0]}. Skipping video {video_name}.")
                    return
            except Exception as e:
                print(f"Error reading first frame {first_frame_path[0]} for video {video_name}: {e}. Skipping.")
                return

            height, width, _ = first_frame.shape
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed codec
                output_video_path = output_video_path.replace('.mp4', '.mp4')  # Ensure .mp4 extension
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
                
                if not out.isOpened():
                    print(f"Error: Could not open video writer for {output_video_path}. Skipping this video.")
                    return
            
            except Exception as e:
                print(f"Error initializing video writer for {output_video_path}: {e}. Skipping.")
                return


            frame_files = sorted([f for f in os.listdir(video_frames_path) if f.endswith('.jpg') or f.endswith('.png')])


            for frame_file in frame_files:
                frame_path = os.path.join(video_frames_path, frame_file)
                pose_file = frame_file.replace('.jpg', '.json').replace('.png', '.json')
                pose_path = os.path.join(actual_poses_path, pose_file)

                try:
                    image = cv2.imread(frame_path)
                    if image is None:
                        print(f"Warning: Could not read image {frame_path}. Skipping frame.")
                        continue
                except Exception as e:
                    print(f"Error reading image {frame_path}: {e}. Skipping frame.")
                    continue

                # Load pose landmarks
                landmarks = None
                if os.path.exists(pose_path):
                    try:
                        with open(pose_path, 'r') as f:
                            pose_data = json.load(f)
                            if pose_data:
                                landmarks = pose_data
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {pose_path}: {e}. Skipping landmarks for this frame.")
                    except Exception as e:
                        print(f"Error loading pose data from {pose_path}: {e}. Skipping landmarks for this frame.")

                # Overlay landmarks if available
                if landmarks:
                    try:
                        # Reconstruct NormalizedLandmarkList from the loaded JSON data
                        if isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], dict):
                            normalized_landmarks = landmark_pb2.NormalizedLandmarkList()
                            for lm in landmarks:
                                normalized_landmark = normalized_landmarks.landmark.add()
                                normalized_landmark.x = lm.get('x', 0.0)
                                normalized_landmark.y = lm.get('y', 0.0)
                                normalized_landmark.z = lm.get('z', 0.0)
                                normalized_landmark.visibility = lm.get('visibility', 0.0)
                            
                            mp_drawing.draw_landmarks(image, normalized_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=4),
                                                         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4))
                        else:
                            print(f"Warning: Malformed landmark data for {pose_path}. Expected list of dicts.")
                    except Exception as e:
                        print(f"Error overlaying landmarks for {pose_path}: {e}. Skipping landmark drawing for this frame.")


                # Get feedback from feedback_analyzer.py
                # Get feedback from feedback_analyzer.py
                # The all_video_results is a dictionary with video_name as key and a dictionary of results as value.
                # The inner dictionary contains 'per_frame_feedback' and 'phase_data'.
                # 'per_frame_feedback' is a list of lists, where each inner list contains feedback strings for a frame.
                # 'phase_data' is a dictionary containing phase names and their corresponding frame intervals.

                per_frame_feedback_list = current_video_feedback_data.get('per_frame_feedback', [])
                phase_data = current_video_feedback_data.get('phase_data', {})

                # Extract frame index from filename
                # Assuming frame filenames are like 'frame_0001.jpg'
                try:
                    # Handle both "frame_0001.jpg" and "0001.jpg" formats
                    if '_' in frame_file:
                        frame_index = int(frame_file.split('_')[-1].split('.')[0])
                    else:
                        frame_index = int(frame_file.split('.')[0])
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract frame index from {frame_file}. Skipping feedback for this frame.")
                    frame_index = -1

                # Determine current phase
                current_phase = "N/A"
                if phase_data:
                    for phase, intervals in phase_data.items():
                        if intervals:  # Check if intervals list is not empty
                            for start, end in intervals:
                                if start <= frame_index <= end:
                                    current_phase = phase.replace('_', ' ').title()
                                    break
                            if current_phase != "N/A":
                                break

                # Get feedback text for the current frame with bounds checking
                feedback_text = ""
                # Get feedback text for the current frame with bounds checking
                feedback_text = ""
                if frame_index != -1 and 0 <= frame_index < len(per_frame_feedback_list):
                    frame_feedback_items = per_frame_feedback_list[frame_index]
                    if frame_feedback_items and isinstance(frame_feedback_items, list):
                        feedback_text = "; ".join(frame_feedback_items)
                
                # Overlay phase name

                if current_phase != "N/A":
                    phase_display_text = f"Phase: {current_phase}"
                    (text_width, text_height), baseline = cv2.getTextSize(phase_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(image, (10, 30 - text_height - baseline), (10 + text_width, 30), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, phase_display_text, (10, 30 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Overlay feedback text

                if feedback_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6 # Slightly smaller font
                    font_thickness = 1 # Thinner text
                    text_color = (255, 255, 255)
                    background_color = (0, 0, 0)
                    line_height = int(cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] * 1.5) # Approximate line height
                    max_width = width - 20 # 10 pixels padding on each side

                    words = feedback_text.split(' ')
                    current_line = []
                    lines = []

                    for word in words:
                        test_line = ' '.join(current_line + [word])
                        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
                        if text_width < max_width:
                            current_line.append(word)
                        else:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                    lines.append(' '.join(current_line))

                    # Calculate starting y_offset from the bottom
                    total_text_height = len(lines) * line_height
                    y_offset = height - total_text_height - 10 # 10 pixels from the bottom

                    for i, line in enumerate(lines):
                        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
                        # Draw background rectangle for the line
                        cv2.rectangle(image, (10, y_offset - text_height - baseline), (10 + text_width, y_offset), background_color, cv2.FILLED)
                        # Put the text on the image
                        cv2.putText(image, line, (10, y_offset - baseline), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        y_offset += line_height # Move to the next line position

                # Overlay frame number and timecode
                time_in_seconds = frame_index / fps
                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)
                milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)
                timecode_text = f"Frame: {frame_index} | Time: {minutes:02d}:{seconds:02d}:{milliseconds:03d}"
                
                (tc_text_width, tc_text_height), tc_baseline = cv2.getTextSize(timecode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tc_x = width - tc_text_width - 10
                tc_y = height - 10
                cv2.rectangle(image, (tc_x - 5, tc_y - tc_text_height - tc_baseline - 5), (tc_x + tc_text_width + 5, tc_y + 5), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, timecode_text, (tc_x, tc_y - tc_baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                try:
                    out.write(image)
                except Exception as e:
                    print(f"Error writing frame {frame_path} to video for {video_name}: {e}. Continuing to next frame.")



            out.release()


if __name__ == '__main__':
    # Define paths relative to the script location
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(script_dir) # Go up one level from src

    frames_directory = os.path.join(base_dir, 'frames')
    poses_directory = os.path.join(base_dir, 'poses')
    output_directory = os.path.join(base_dir, 'videos')

    # Run feedback analysis to get per-frame feedback
    from feedback_analyzer import process_poses_directory
    all_video_analysis_results = process_poses_directory(poses_directory)

    visualize_feedback(frames_directory, poses_directory, output_directory, all_video_analysis_results)