import os
import cv2

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extracts frames from a video and saves them as JPEG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
        frame_interval (int): Interval at which to extract frames (e.g., 5 for every 5th frame).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()



if __name__ == '__main__':
    videos_dir = "/Users/kartikgarg/Downloads/futuresportler_archery/videos"
    frames_dir = "/Users/kartikgarg/Downloads/futuresportler_archery/frames"

    for video_file in os.listdir(videos_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(videos_dir, video_file)
            extract_frames(video_path, frames_dir)