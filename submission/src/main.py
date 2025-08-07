import os
import cv2
import json
import shutil
import os
from tqdm import tqdm

from pose_estimator import estimate_pose, save_landmarks_to_csv
from feedback_analyzer import process_poses_directory
from visualizer import visualize_feedback
from video_utils import extract_frames

# Get absolute paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

VIDEO_DIR = os.path.join(project_root, 'videos')
OUTPUT_DIR = os.path.join(project_root, 'frames')
LANDMARKS_DIR = os.path.join(project_root, 'landmarks', 'csv_landmarks')
POSES_DIR = os.path.join(project_root, 'poses')
FEEDBACK_DIR = os.path.join(project_root, 'feedback')
FEEDBACK_VIDEO_DIR = os.path.join(project_root, 'output_videos')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    os.makedirs(POSES_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_VIDEO_DIR, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

    # Step 1: Extract frames and detect poses for all videos
    for video_file in tqdm(video_files, desc="Extracting frames and detecting poses"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        current_frames_dir = os.path.join(OUTPUT_DIR, video_name)
        current_poses_dir = os.path.join(POSES_DIR, video_name)
        
        os.makedirs(current_frames_dir, exist_ok=True)
        os.makedirs(current_poses_dir, exist_ok=True)
        
        # Extract frames
        extract_frames(video_path, current_frames_dir)
        
        # Detect poses
        estimate_pose(current_frames_dir, current_poses_dir)

    # Step 2: Save landmarks to CSV
    save_landmarks_to_csv(POSES_DIR, LANDMARKS_DIR)

    # Step 3: Analyze posture for all videos
    all_video_results = process_poses_directory(POSES_DIR, FEEDBACK_DIR)
    
    # Save individual feedback JSON files
    for video_name, feedback_data in all_video_results.items():
        individual_feedback_json_path = os.path.join(FEEDBACK_DIR, f"{video_name}_feedback.json")
        with open(individual_feedback_json_path, 'w') as f:
            json.dump(feedback_data, f, indent=4)



    # Step 4: Generate feedback videos
    for video_file in tqdm(video_files, desc="Generating feedback videos"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Get FPS for this video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Get feedback data for current video
        current_video_feedback_data = all_video_results.get(video_name, {})
        

        
        # Generate feedback video
        visualize_feedback(
            video_name,
            OUTPUT_DIR,
            POSES_DIR,
            FEEDBACK_VIDEO_DIR,
            current_video_feedback_data,
            fps
        )
        


    print("Processing complete. Feedback videos and JSONs generated.")

if __name__ == '__main__':
    main()