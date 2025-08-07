# FutureSportler Archery Analysis Project

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
The core of the project is orchestrated by `main.py`, which integrates functionalities from other modules:
1.  **`video_utils.py`**: Handles the extraction of individual frames from input videos.
2.  **`pose_estimator.py`**: Detects pose landmarks on each extracted frame and saves them as JSON files. It also includes functionality to convert these JSON pose data into CSV files for each video, located in `outputs/csv_landmarks/`.
3.  **`feedback_analyzer.py`**: Processes the pose data to calculate angles (e.g., elbow angles), analyze posture, and generate textual feedback. It also calculates consistency and symmetry scores for key body parts.
4.  **`visualizer.py`**: Overlays the detected landmarks and the generated feedback text onto the original video frames, creating a new video with visual corrections and insights.

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
To run the entire analysis pipeline, execute the `main.py` script from the project root directory:

```bash
python3 src/main.py
```

This script will:
-   Read videos from the `videos/` directory.
-   Extract frames into the `frames/` directory.
-   Detect poses and save JSON data into the `poses/` directory.
-   Generate CSV files of pose keypoints in `outputs/csv_landmarks/`.
-   Analyze posture and calculate feedback scores.
-   Create feedback videos with overlaid corrections in the `outputs/` directory.

### Project Submission Includes:
-   **5 Processed Videos**: The `outputs/` directory contains feedback videos for `archery_1.mp4` through `archery_5.mp4`.
-   **Pose Keypoints**: Raw pose landmark data saved as JSON files in `poses/`.
-   **Feedback Scores**: Consistency and symmetry scores for elbow angles are calculated and displayed during the analysis phase.
-   **Optional 3D CSV**: CSV files containing pose keypoint coordinates for each frame, suitable for 3D visualization in tools like Blender, are generated in `outputs/csv_landmarks/`.

This project aims to automate and optimize the job search and application process for users by intelligently analyzing resumes, matching them to job postings, and executing applications.