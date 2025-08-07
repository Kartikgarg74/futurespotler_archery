# Archery Feedback System

This project provides an automated system for analyzing archery form from video, estimating human pose, and generating visual feedback with overlaid phase names, feedback statements, frame numbers, and timecodes.

## Project Structure

- `videos/`: Contains the final output videos with overlaid feedback.
- `landmarks/`: Stores the saved landmark data (CSV files) for each video frame.
- `feedback/`: Contains per-phase feedback in JSON format.
- `src/`: Holds the Python source code files for pose estimation, feedback analysis, and visualization.
- `README.md`: This file, summarizing the project.

## Implemented Features

1.  **Pose Estimation**: Utilizes MediaPipe Pose to detect and track key body landmarks in each video frame.
2.  **Feedback Analysis**: Analyzes the estimated poses to identify specific form issues during different archery phases.
3.  **Dynamic Visualization**: Overlays the following information on each frame of the output video:
    -   Current detected phase name (e.g., 'Draw Phase').
    -   Relevant feedback statement (e.g., 'Elbow too low').
    -   Current frame number.
    -   Formatted timecode (minutes:seconds:milliseconds).
    All overlaid text includes a black background for improved readability.

## How it Works

The `main.py` script orchestrates the entire process:

1.  Reads input videos.
2.  Extracts frames and performs pose estimation.
3.  Saves landmark data to CSV files.
4.  Analyzes posture and generates feedback based on predefined rules.
5.  Uses the `visualizer.py` module to create output videos with all the specified overlays.

## Usage

To run the system, execute `main.py`. Ensure all necessary dependencies are installed.

```bash
python src/main.py
```

The output videos, landmark data, and feedback JSONs will be organized into their respective directories within the `submission/` folder.