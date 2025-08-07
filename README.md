# Archery Posture Analysis and Feedback System

## Project Purpose
This project aims to provide an automated system for analyzing archery posture and providing feedback. It processes video footage of archers, extracts key pose landmarks, analyzes their movements, and generates visual feedback to help improve technique.

## Task Assigned by FutureSportler
The primary task was to develop a pipeline that can:
1.  Extract frames from each video.
2.  Detect poses on each frame.
3.  Analyze posture based on detected poses.
4.  Overlay corrections and feedback onto the video frames.
5.  Export the final processed video with feedback.

Additionally, the project was enhanced to include a scoring system for movement phases (though not explicitly detected in this version, consistency and symmetry are scored), and to export pose keypoints in a CSV format for potential 3D visualization.

## Tools Used
-   **MediaPipe Pose**: For robust and accurate pose detection.
-   **OpenCV (cv2)**: For video processing, frame extraction, and overlaying visuals.
-   **NumPy**: For numerical operations, especially in angle calculations and score aggregation.
-   **Python's `os` and `json` modules**: For file system operations and handling pose data.

## How the Pipeline Works
The core of the project is orchestrated by <mcfile name="main.py" path="submission/src/main.py"></mcfile>, which integrates functionalities from other modules:
1.  <mcfile name="video_utils.py" path="submission/src/video_utils.py"></mcfile>: Handles the extraction of individual frames from input videos.
2.  <mcfile name="pose_estimator.py" path="submission/src/pose_estimator.py"></mcfile>: Detects pose landmarks on each extracted frame and saves them as JSON files. It also includes functionality to convert these JSON pose data into CSV files for each video, located in `submission/landmarks/csv_landmarks/`.
3.  <mcfile name="feedback_analyzer.py" path="submission/src/feedback_analyzer.py"></mcfile>: Processes the pose data to calculate angles (e.g., elbow angles), analyze posture, and generate textual feedback. It now includes comprehensive analysis for the **stance phase**, evaluating body alignment (shoulders, hips, ankles) and foot placement. It also calculates consistency and symmetry scores for key body parts.
4.  <mcfile name="visualizer.py" path="submission/src/visualizer.py"></mcfile>: Overlays the detected landmarks and the generated feedback text onto the original video frames, creating a new video with visual corrections and insights.

## Instructions to Run the Full Project Locally

### Prerequisites
-   Python 3.8+
-   `pip` (Python package installer)

### Setup
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd futuresportler_archery
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate the virtual environment**:
    -   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline
To run the entire analysis pipeline, execute the <mcfile name="main.py" path="submission/src/main.py"></mcfile> script from the project root directory:

```bash
python3 submission/src/main.py
```

This script will:
-   Read videos from the `videos/` directory.
-   Extract frames into the `submission/frames/` directory.
-   Detect poses and save JSON data into the `submission/poses/` directory.
-   Generate CSV files of pose keypoints in `submission/landmarks/csv_landmarks/`.
-   Analyze posture and calculate feedback scores, including detailed stance analysis.
-   **Save individual feedback JSON files** for each video in `submission/feedback/`.
-   Create feedback videos with overlaid corrections in the `submission/output_videos/` directory.

### Project Submission Includes:
-   **5 Processed Videos**: The `submission/output_videos/` directory contains feedback videos for `archery_1.mp4` through `archery_5.mp4`.
-   **Pose Keypoints**: Raw pose landmark data saved as JSON files in `submission/poses/`.
-   **Feedback JSONs**: Individual feedback JSON files for each video are saved in `submission/feedback/`.
-   **Feedback Scores**: Consistency and symmetry scores for elbow angles and detailed stance analysis are calculated and displayed during the analysis phase.
-   **Optional 3D CSV**: CSV files containing pose keypoint coordinates for each frame, suitable for 3D visualization in tools like Blender, are generated in `submission/landmarks/csv_landmarks/`.

This project aims to automate and optimize the job search and application process for users by intelligently analyzing resumes, matching them to job postings, and executing applications.