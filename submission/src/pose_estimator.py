import os
import json
import cv2
import mediapipe as mp
import os
import csv

def estimate_pose(frames_folder, poses_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)

    os.makedirs(poses_folder, exist_ok=True)
    processed_frames_count = 0
    frame_files = [f for f in sorted(os.listdir(frames_folder)) if f.endswith('.jpg')]

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Could not read image {frame_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks_data = []
            for landmark in results.pose_landmarks.landmark:
                landmarks_data.append({
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                })
            
            # Save landmarks to a JSON file
            frame_base_name = os.path.splitext(frame_file)[0]
            json_output_path = os.path.join(poses_folder, f'{frame_base_name}.json')
            with open(json_output_path, 'w') as f:
                json.dump(landmarks_data, f, indent=4)
            processed_frames_count += 1
        else:
            print(f"No pose landmarks detected for {frame_path}. Skipping JSON creation for this frame.")
            # Optionally, save an empty JSON or a JSON indicating no landmarks
            # For now, we just skip saving the JSON if no landmarks are detected
            continue




    pose.close()

def save_landmarks_to_csv(poses_folder, output_csv_folder):
    """
    Converts pose landmark JSON files into a single CSV file for each video.
    Each row in the CSV represents a frame, and columns represent keypoint coordinates.
    """
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)

    for video_name in os.listdir(poses_folder):
        video_poses_path = os.path.join(poses_folder, video_name)
        if os.path.isdir(video_poses_path):
            csv_output_path = os.path.join(output_csv_folder, f'{video_name}_landmarks.csv')
            
            all_landmarks_data = []
            header = []

            frame_files = sorted([f for f in os.listdir(video_poses_path) if f.endswith('.json')])
            for frame_file in frame_files:
                json_path = os.path.join(video_poses_path, frame_file)
                with open(json_path, 'r') as f:
                    landmarks = json.load(f)
                
                row_data = {}
                if not header:
                    # Create header from the first frame's landmarks
                    for i, landmark in enumerate(landmarks):
                        row_data[f'landmark_{i}_x'] = landmark['x']
                        row_data[f'landmark_{i}_y'] = landmark['y']
                        row_data[f'landmark_{i}_z'] = landmark['z']
                        row_data[f'landmark_{i}_visibility'] = landmark['visibility']
                    header = list(row_data.keys())
                else:
                    for i, landmark in enumerate(landmarks):
                        row_data[f'landmark_{i}_x'] = landmark['x']
                        row_data[f'landmark_{i}_y'] = landmark['y']
                        row_data[f'landmark_{i}_z'] = landmark['z']
                        row_data[f'landmark_{i}_visibility'] = landmark['visibility']
                all_landmarks_data.append(row_data)

            if all_landmarks_data:
                with open(csv_output_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(all_landmarks_data)
                print(f"Saved CSV for {video_name} to {csv_output_path}")

# if __name__ == '__main__':
    # The main execution block for pose_estimator.py is typically for testing.
    # In the main application, estimate_pose and save_landmarks_to_csv are called from main.py.
    # If you need to test this script independently, uncomment and adjust the paths below.
    # current_dir = os.path.dirname(__file__)
    # project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # frames_dir = os.path.join(project_root, 'frames', 'archery_4') # Example: specify a video's frame directory
    # poses_dir = os.path.join(project_root, 'poses', 'archery_4') # Example: specify a video's pose directory

    # os.makedirs(frames_dir, exist_ok=True)
    # os.makedirs(poses_dir, exist_ok=True)

    # estimate_pose(frames_dir, poses_dir)

    # output_csv_dir = os.path.join(project_root, 'landmarks', 'csv_landmarks')
    # save_landmarks_to_csv(poses_dir, output_csv_dir)