import os
import json
import numpy as np
import os

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points (p1-p2-p3) in 2D or 3D.
    The angle is at p2.
    """
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0 # Avoid division by zero

    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_symmetry_score(left_angle, right_angle, ideal_diff_threshold=5):
    """
    Calculates a symmetry score (out of 10) based on the difference between left and right angles.
    Lower difference means higher score.
    """
    diff = abs(left_angle - right_angle)
    if diff <= ideal_diff_threshold:
        return 10.0
    else:
        # Linear decay: score drops by 1 for every 'ideal_diff_threshold' increase in diff
        score = 10.0 - (diff - ideal_diff_threshold) / ideal_diff_threshold
        return max(0.0, score)

def calculate_consistency_score(angles, ideal_std_dev_threshold=5):
    """
    Calculates a consistency score (out of 10) based on the standard deviation of angles.
    Lower standard deviation means higher score.
    """
    if not angles or len(angles) < 2:
        return 0.0 # Not enough data to calculate consistency
    std_dev = np.std(angles)
    if std_dev <= ideal_std_dev_threshold:
        return 10.0
    else:
        # Linear decay: score drops by 1 for every 'ideal_std_dev_threshold' increase in std_dev
        score = 10.0 - (std_dev - ideal_std_dev_threshold) / ideal_std_dev_threshold
        return max(0.0, score)

def analyze_pose(pose_data):
    """
    Analyzes pose data to provide feedback and extract key angles for scoring.
    pose_data is a list of dictionaries, where each dictionary represents a landmark.
    """
    feedback = []
    angles = {}

    # MediaPipe Pose landmark indices:
    # 11: left_shoulder, 13: left_elbow, 15: left_wrist
    # 12: right_shoulder, 14: right_elbow, 16: right_wrist
    # 23: left_hip, 24: right_hip
    # 25: left_knee, 26: right_knee
    # 27: left_ankle, 28: right_ankle

    if len(pose_data) > 28: # Ensure enough landmarks exist for common calculations
        # Elbow angles
        left_shoulder = pose_data[11]
        left_elbow = pose_data[13]
        left_wrist = pose_data[15]
        right_shoulder = pose_data[12]
        right_elbow = pose_data[14]
        right_wrist = pose_data[16]

        elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

        angles['elbow_left'] = elbow_angle_left
        angles['elbow_right'] = elbow_angle_right

        if elbow_angle_left < 160: # Example threshold
            feedback.append(f"Left elbow angle ({elbow_angle_left:.2f} degrees) is too bent. Aim for straighter arm.")
        else:
            feedback.append(f"Left elbow angle ({elbow_angle_left:.2f} degrees) looks good.")

        if elbow_angle_right < 160: # Example threshold
            feedback.append(f"Right elbow angle ({elbow_angle_right:.2f} degrees) is too bent. Aim for straighter arm.")
        else:
            feedback.append(f"Right elbow angle ({elbow_angle_right:.2f} degrees) looks good.")

        # Add more angle calculations as needed for other joints (e.g., shoulder, knee, hip)
        # For example, shoulder angle (torso-shoulder-elbow)
        # left_hip = pose_data[23]
        # left_shoulder = pose_data[11]
        # left_elbow = pose_data[13]
        # shoulder_angle_left = calculate_angle(left_hip, left_shoulder, left_elbow)
        # angles['shoulder_left'] = shoulder_angle_left

    return feedback, angles

def detect_phases(all_frames_pose_data):

    phases = {
        'stance': [],
        'setup': [],
        'draw': [],
        'anchor': [],
        'release': [],
        'follow_through': []
    }

    # Heuristics for phase detection (simplified examples):
    # These will need to be refined based on actual data and observation.
    # Frame indices are 0-based.

    prev_pose_data = None # To store pose data from the previous frame for velocity calculation
    frame_velocities = {}

    for i, pose_data in enumerate(all_frames_pose_data):
        if len(pose_data) > 28: # Ensure enough landmarks are detected
            # Right wrist x-coordinate as a proxy for arm extension
            right_wrist_x = pose_data[16]['x']
            right_shoulder_x = pose_data[12]['x']

            left_wrist_y = pose_data[15]['y']
            right_wrist_y = pose_data[16]['y']
            left_shoulder_y = pose_data[11]['y']
            right_shoulder_y = pose_data[12]['y']

            # Initialize distance_nose_wrist with a default value
            distance_nose_wrist = 1.0 # Default to a large value if landmarks are not found

            if len(pose_data) > 16 and len(pose_data) > 0: # Ensure landmarks exist before calculating distance
                nose = pose_data[0]
                right_wrist = pose_data[16]
                distance_nose_wrist = np.sqrt((nose['x'] - right_wrist['x'])**2 + (nose['y'] - right_wrist['y'])**2)

            # Velocity calculations (if previous frame data is available)
            if prev_pose_data and len(prev_pose_data) > 28:
                # Right wrist velocity
                prev_right_wrist = prev_pose_data[16]
                right_wrist_velocity_x = right_wrist_x - prev_right_wrist['x']
                right_wrist_velocity_y = pose_data[16]['y'] - prev_right_wrist['y']
                right_wrist_speed = np.sqrt(right_wrist_velocity_x**2 + right_wrist_velocity_y**2)

                # Left wrist (bow hand) velocity
                prev_left_wrist = prev_pose_data[15]
                left_wrist_velocity_x = pose_data[15]['x'] - prev_left_wrist['x']
                left_wrist_velocity_y = pose_data[15]['y'] - prev_left_wrist['y']
                left_wrist_speed = np.sqrt(left_wrist_velocity_x**2 + left_wrist_velocity_y**2)

                # Right elbow velocity
                prev_right_elbow = prev_pose_data[14]
                right_elbow_velocity_x = pose_data[14]['x'] - prev_right_elbow['x']
                right_elbow_velocity_y = pose_data[14]['y'] - prev_right_elbow['y']
                right_elbow_speed = np.sqrt(right_elbow_velocity_x**2 + right_elbow_velocity_y**2)
            else:
                right_wrist_speed = 0
                left_wrist_speed = 0
                right_elbow_speed = 0

            # Heuristic for 'setup' phase: wrists are roughly around shoulder height or slightly below
            if (left_wrist_y > left_shoulder_y - 0.05) and (left_wrist_y < left_shoulder_y + 0.1) and \
               (right_wrist_y > right_shoulder_y - 0.05) and (right_wrist_y < right_shoulder_y + 0.1):
                phases['setup'].append(i)
            # Heuristic for 'anchor' phase: right wrist close to nose
            elif distance_nose_wrist < 0.15: # Adjusted threshold for close proximity
                phases['anchor'].append(i)
            elif right_wrist_x < right_shoulder_x - 0.1: # Arbitrary threshold for draw
                phases['draw'].append(i)
            # Heuristic for 'anchor' phase: right wrist close to nose
            elif distance_nose_wrist < 0.15: # Adjusted threshold for close proximity
                phases['anchor'].append(i)
            # Heuristic for 'release' phase: right wrist moves significantly to the right after anchor
            # Now incorporating velocity for more precise detection
            elif right_wrist_x > right_shoulder_x + 0.1 and distance_nose_wrist > 0.2 and right_wrist_speed > 0.01: # Arbitrary thresholds, added speed
                phases['release'].append(i)
            # Heuristic for 'follow_through' phase: right wrist continues to move right
            elif right_wrist_x > right_shoulder_x + 0.2: # Arbitrary threshold for follow-through
                phases['follow_through'].append(i)
            else:
                phases['stance'].append(i)

            # Store velocities for the current frame
            frame_velocities[i] = {
                'right_wrist_speed': right_wrist_speed,
                'left_wrist_speed': left_wrist_speed,
                'right_elbow_speed': right_elbow_speed
            }
        else:
            # If not enough landmarks, assign default (zero) velocities
            frame_velocities[i] = {
                'right_wrist_speed': 0,
                'left_wrist_speed': 0,
                'right_elbow_speed': 0
            }

        prev_pose_data = pose_data # Store current pose data for the next iteration

    # Further refinement needed to define clear start/end of phases
    # This simple example will likely have overlapping or incorrect phases.

    def group_consecutive_frames(frame_indices):
        if not frame_indices:
            return []

        frame_indices.sort()
        grouped_intervals = []
        current_interval_start = frame_indices[0]
        current_interval_end = frame_indices[0]

        for i in range(1, len(frame_indices)):
            if frame_indices[i] == current_interval_end + 1:
                current_interval_end = frame_indices[i]
            else:
                grouped_intervals.append([current_interval_start, current_interval_end])
                current_interval_start = frame_indices[i]
                current_interval_end = frame_indices[i]

        grouped_intervals.append([current_interval_start, current_interval_end])
        return grouped_intervals

    # Convert raw frame indices into grouped intervals for each phase
    for phase_name, frame_list in phases.items():
        phases[phase_name] = group_consecutive_frames(frame_list)

    return phases, frame_velocities
    # A more robust solution would involve state machines, velocity analysis, or ML models.

    # Convert lists of frame indices to start-end tuples
    # This is a very basic conversion and assumes contiguous frames for a phase.
    # A real implementation would group consecutive frames.
    segmented_phases = {}
    for phase_name, frame_indices in phases.items():
        if frame_indices:
            # For simplicity, just taking min/max for now. Real segmentation is harder.
            segmented_phases[phase_name] = [(min(frame_indices), max(frame_indices))]
        else:
            segmented_phases[phase_name] = []
    print(f"Debug: Segmented phases: {segmented_phases}")

    return segmented_phases, frame_velocities

def process_poses_directory(poses_root_folder, feedback_output_dir):
    all_video_results = {}
    for video_name in os.listdir(poses_root_folder):
        video_poses_path = os.path.join(poses_root_folder, video_name)
        if os.path.isdir(video_poses_path) and '_feedback' not in video_name:
            video_feedback = {
                'phase_summaries': {},
                'phase_scores': {},
                'improvement_points': []
            }

            video_angles = {'elbow_left': [], 'elbow_right': []}
            # Add more angle lists here as you add more angle calculations in analyze_pose

            json_files = sorted([f for f in os.listdir(video_poses_path) if f.endswith('.json')])
            if not json_files:
                continue

            all_frames_pose_data = []
            for json_file in json_files:
                json_path = os.path.join(video_poses_path, json_file)
                with open(json_path, 'r') as f:
                    pose_data = json.load(f)
                    all_frames_pose_data.append(pose_data)
            per_frame_feedback = [[] for _ in range(len(all_frames_pose_data))]
            
            # Detect phases and get velocities
            segmented_phases, frame_velocities = detect_phases(all_frames_pose_data)
            video_feedback['overall_feedback'] = "\n--- Phase Analysis ---"
            video_scores = {}

            for phase_name, segments in segmented_phases.items():
                if segments:
                    for start_frame, end_frame in segments:
                        phase_pose_data = all_frames_pose_data[start_frame : end_frame + 1]
                        phase_feedback = []
                        phase_angles = {'elbow_left': [], 'elbow_right': []}

                        if phase_name == 'stance':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            # Existing stance analysis code
                            # ...
                            if phase_pose_data:
                                # Analyze body alignment (shoulders, hips, ankles)
                                # Check for vertical alignment of shoulder, hip, and ankle on both sides
                                feedback_messages = []
                                scores = []

                                for side in ['left', 'right']:
                                    shoulder_idx = 11 if side == 'left' else 12
                                    hip_idx = 23 if side == 'left' else 24
                                    ankle_idx = 27 if side == 'left' else 28

                                    shoulder_y_coords = [p[shoulder_idx]['y'] for p in phase_pose_data if len(p) > shoulder_idx]
                                    hip_y_coords = [p[hip_idx]['y'] for p in phase_pose_data if len(p) > hip_idx]
                                    ankle_y_coords = [p[ankle_idx]['y'] for p in phase_pose_data if len(p) > ankle_idx]

                                    if shoulder_y_coords and hip_y_coords and ankle_y_coords:
                                        # Simple check: average y-coordinates should be somewhat aligned
                                        avg_shoulder_y = np.mean(shoulder_y_coords)
                                        avg_hip_y = np.mean(hip_y_coords)
                                        avg_ankle_y = np.mean(ankle_y_coords)

                                        # Calculate vertical deviation
                                        vertical_deviation = np.std([avg_shoulder_y, avg_hip_y, avg_ankle_y])

                                        if vertical_deviation > 0.03: # Arbitrary threshold for poor alignment
                                            msg = f"Poor {side} side vertical alignment in stance. Deviation: {vertical_deviation:.3f}. Ensure shoulder, hip, and ankle are stacked vertically."
                                            feedback_messages.append(msg)
                                            scores.append(max(0, 10 - (vertical_deviation * 100)))
                                        else:
                                            msg = f"Good {side} side vertical alignment in stance."
                                            feedback_messages.append(msg)
                                            scores.append(10.0)
                                    else:
                                        feedback_messages.append(f"Insufficient {side} side data for vertical alignment analysis.")
                                        scores.append(5.0) # Neutral score if data is missing

                                if feedback_messages:
                                    overall_feedback_msg = " ".join(feedback_messages)
                                    phase_feedback.append(overall_feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        if frame_idx < len(per_frame_feedback):
                                            per_frame_feedback[frame_idx].append(overall_feedback_msg)
                                    video_feedback['phase_summaries'][f'stance_alignment_frames_{start_frame}-{end_frame}'] = overall_feedback_msg
                                    video_scores[f'stance_alignment_score_frames_{start_frame}-{end_frame}'] = np.mean(scores)
                                    if np.mean(scores) < 10.0:
                                        video_feedback['improvement_points'].append(f"Stance: {overall_feedback_msg}")

                                # Check foot placement (distance between ankles)
                                left_ankle_x_coords = [p[27]['x'] for p in phase_pose_data if len(p) > 27]
                                right_ankle_x_coords = [p[28]['x'] for p in phase_pose_data if len(p) > 28]

                                if left_ankle_x_coords and right_ankle_x_coords:
                                    avg_left_ankle_x = np.mean(left_ankle_x_coords)
                                    avg_right_ankle_x = np.mean(right_ankle_x_coords)
                                    foot_distance = abs(avg_left_ankle_x - avg_right_ankle_x)

                                    # Assuming a typical stance width, adjust thresholds as needed
                                    if foot_distance < 0.05: # Too narrow
                                        msg = f"Stance too narrow. Foot distance: {foot_distance:.3f}. Widen your stance for better stability."
                                        phase_feedback.append(msg)
                                        video_feedback['improvement_points'].append(f"Stance: {msg}")
                                    elif foot_distance > 0.2: # Too wide
                                        msg = f"Stance too wide. Foot distance: {foot_distance:.3f}. Bring your feet closer for optimal balance."
                                        phase_feedback.append(msg)
                                        video_feedback['improvement_points'].append(f"Stance: {msg}")
                                    else:
                                        msg = f"Good foot placement in stance. Foot distance: {foot_distance:.3f}."
                                        phase_feedback.append(msg)
                                    video_feedback['phase_summaries'][f'stance_foot_placement_frames_{start_frame}-{end_frame}'] = msg
                                    video_scores[f'stance_foot_placement_score_frames_{start_frame}-{end_frame}'] = 10.0 if 0.05 <= foot_distance <= 0.2 else 5.0
                                else:
                                    phase_feedback.append("Insufficient data for foot placement analysis.")
                                    video_scores[f'stance_foot_placement_score_frames_{start_frame}-{end_frame}'] = 5.0
                        elif phase_name == 'setup':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                        elif phase_name == 'anchor':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                        elif phase_name == 'release':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            if phase_pose_data:
                                # Analyze right wrist velocity for smoothness
                                release_right_wrist_speeds = [frame_velocities[frame_idx]['right_wrist_speed'] for frame_idx in range(start_frame, end_frame + 1)]
                                if release_right_wrist_speeds:
                                    avg_right_wrist_speed = np.mean(release_right_wrist_speeds)
                                    # Simple check for jerkiness: high standard deviation or sudden drops/spikes
                                    std_right_wrist_speed = np.std(release_right_wrist_speeds)

                                    if std_right_wrist_speed > 0.015: # Arbitrary threshold for jerkiness
                                        feedback_msg = f"Release too jerky, right wrist speed variation: {std_right_wrist_speed:.3f}. Aim for a smoother, more consistent movement."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            if frame_idx < len(per_frame_feedback):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                        smoothness_score = max(0, 10 - (std_right_wrist_speed * 100)) # Example scoring
                                    else:
                                        feedback_msg = "Release appears smooth and controlled."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            if frame_idx < len(per_frame_feedback):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                        smoothness_score = 10.0
                                    video_scores[f'release_smoothness_score_frames_{start_frame}-{end_frame}'] = smoothness_score
                                    video_feedback['phase_summaries'][f'release_frames_{start_frame}-{end_frame}'] = phase_feedback[0]
                                    if smoothness_score < 10.0:
                                        video_feedback['improvement_points'].append(f"Release: {phase_feedback[0]}")

                                # Analyze left wrist (bow hand) velocity for flinch
                                release_left_wrist_speeds = [frame_velocities[frame_idx]['left_wrist_speed'] for frame_idx in range(start_frame, end_frame + 1)]
                                if release_left_wrist_speeds:
                                    max_left_wrist_speed = np.max(release_left_wrist_speeds)
                                    if max_left_wrist_speed > 0.01: # Arbitrary threshold for flinch
                                        feedback_msg = f"Bow hand flinch detected. Max left wrist movement: {max_left_wrist_speed:.3f} px/frame. Keep the bow hand still."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            if frame_idx < len(per_frame_feedback):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Release: Bow hand flinch detected. Max left wrist movement: {max_left_wrist_speed:.3f} px/frame. Keep the bow hand still.")
                                    else:
                                        feedback_msg = "Bow hand remained stable during release."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            if frame_idx < len(per_frame_feedback):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                    video_feedback['phase_summaries'][f'release_bow_hand_frames_{start_frame}-{end_frame}'] = phase_feedback[-1]

                        elif phase_name == 'draw':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")

                        elif phase_name == 'follow_through':
                            for frame_idx in range(start_frame, end_frame + 1):
                                if frame_idx < len(per_frame_feedback):
                                    per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            if phase_pose_data:
                                # Analyze body posture stability (e.g., using hip and shoulder consistency)
                                left_hip_y_coords = [p[23]['y'] for p in phase_pose_data if len(p) > 23]
                                right_hip_y_coords = [p[24]['y'] for p in phase_pose_data if len(p) > 24]
                                left_shoulder_y_coords = [p[11]['y'] for p in phase_pose_data if len(p) > 11]
                                right_shoulder_y_coords = [p[12]['y'] for p in phase_pose_data if len(p) > 12]

                                if left_hip_y_coords and right_hip_y_coords and left_shoulder_y_coords and right_shoulder_y_coords:
                                    hip_y_std = np.std(left_hip_y_coords + right_hip_y_coords)
                                    shoulder_y_std = np.std(left_shoulder_y_coords + right_shoulder_y_coords)

                                    if hip_y_std > 0.02 or shoulder_y_std > 0.02: # Arbitrary threshold for instability
                                        feedback_msg = f"Body posture unstable during follow-through. Hip variation: {hip_y_std:.3f}, Shoulder variation: {shoulder_y_std:.3f}. Consider practicing a stable hold after release."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Follow-through: {feedback_msg}")
                                    else:
                                        feedback_msg = "Body posture remained stable during follow-through."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    video_feedback['phase_summaries'][f'follow_through_posture_frames_{start_frame}-{end_frame}'] = feedback_msg

                                # Analyze head movement (nose landmark)
                                nose_x_coords = [p[0]['x'] for p in phase_pose_data if len(p) > 0]
                                nose_y_coords = [p[0]['y'] for p in phase_pose_data if len(p) > 0]

                                if len(nose_x_coords) > 1 and len(nose_y_coords) > 1:
                                    nose_movement_x_std = np.std(nose_x_coords)
                                    nose_movement_y_std = np.std(nose_y_coords)

                                    if nose_movement_x_std > 0.01 or nose_movement_y_std > 0.01: # Arbitrary threshold for head movement
                                        feedback_msg = f"Head movement detected during follow-through. X variation: {nose_movement_x_std:.3f}, Y variation: {nose_movement_y_std:.3f}. Ensure head remains still or follows the arrow smoothly."
                                        phase_feedback.append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Follow-through: {feedback_msg}")
                                    else:
                                        feedback_msg = "Head remained stable during follow-through."
                                        phase_feedback.append(feedback_msg)
                                    video_feedback['phase_summaries'][f'follow_through_head_frames_{start_frame}-{end_frame}'] = feedback_msg

                                # Analyze hand movement post-release (right wrist speed)
                                follow_through_right_wrist_speeds = [frame_velocities[frame_idx]['right_wrist_speed'] for frame_idx in range(start_frame, end_frame + 1)]
                                if follow_through_right_wrist_speeds:
                                    max_right_wrist_speed_ft = np.max(follow_through_right_wrist_speeds)
                                    if max_right_wrist_speed_ft > 0.01: # Arbitrary threshold for hand movement
                                        feedback_msg = f"Draw hand moved significantly during follow-through. Max right wrist movement: {max_right_wrist_speed_ft:.3f} px/frame. Aim for a calm and controlled hand movement."
                                        phase_feedback.append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Follow-through: {feedback_msg}")
                                    else:
                                        feedback_msg = "Draw hand remained calm and controlled during follow-through."
                                        phase_feedback.append(feedback_msg)
                                    video_feedback['phase_summaries'][f'follow_through_hand_frames_{start_frame}-{end_frame}'] = feedback_msg

                        elif phase_name == 'draw':
                            for frame_idx in range(start_frame, end_frame + 1):
                                per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            if phase_pose_data:
                                # Track movement of string-side elbow (right elbow for right-handed archer)
                                right_elbow_x_coords = [p[14]['x'] for p in phase_pose_data if len(p) > 14]
                                right_elbow_y_coords = [p[14]['y'] for p in phase_pose_data if len(p) > 14]

                                if len(right_elbow_x_coords) > 1 and len(right_elbow_y_coords) > 1:
                                    # Calculate path length and straightness
                                    path_length = 0
                                    for i in range(1, len(right_elbow_x_coords)):
                                        path_length += np.sqrt((right_elbow_x_coords[i] - right_elbow_x_coords[i-1])**2 +
                                                                (right_elbow_y_coords[i] - right_elbow_y_coords[i-1])**2)
                                    
                                    # A very simple measure of straightness: compare path length to direct distance
                                    direct_distance = np.sqrt((right_elbow_x_coords[-1] - right_elbow_x_coords[0])**2 +
                                                              (right_elbow_y_coords[-1] - right_elbow_y_coords[0])**2)
                                    
                                    if path_length > 0 and direct_distance > 0:
                                        straightness_ratio = direct_distance / path_length
                                        if straightness_ratio < 0.9: # Arbitrary threshold for curved path
                                            feedback_msg = f"Draw path is curved. Straightness ratio: {straightness_ratio:.2f}. Aim for a straighter elbow path."
                                            phase_feedback.append(feedback_msg)
                                            for frame_idx in range(start_frame, end_frame + 1):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                            video_feedback['improvement_points'].append(f"Draw: {feedback_msg}")
                                        else:
                                            feedback_msg = "Elbow path during draw is consistent and straight."
                                            phase_feedback.append(feedback_msg)
                                            for frame_idx in range(start_frame, end_frame + 1):
                                                per_frame_feedback[frame_idx].append(feedback_msg)
                                    else:
                                        feedback_msg = "Insufficient movement data for elbow path analysis."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    video_feedback['phase_summaries'][f'draw_elbow_path_frames_{start_frame}-{end_frame}'] = feedback_msg

                                # Check shoulder height consistency (no shoulder drop)
                                left_shoulder_y_coords = [p[11]['y'] for p in phase_pose_data if len(p) > 11]
                                right_shoulder_y_coords = [p[12]['y'] for p in phase_pose_data if len(p) > 12]

                                if len(left_shoulder_y_coords) > 1 and len(right_shoulder_y_coords) > 1:
                                    left_shoulder_y_std = np.std(left_shoulder_y_coords)
                                    right_shoulder_y_std = np.std(right_shoulder_y_coords)

                                    if left_shoulder_y_std > 0.02: # Arbitrary threshold for inconsistency
                                        feedback_msg = f"Left shoulder height inconsistent during draw. Variation: {left_shoulder_y_std:.3f}."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Draw: {feedback_msg}")
                                    else:
                                        feedback_msg = "Left shoulder height is consistent during draw."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    video_feedback['phase_summaries'][f'draw_left_shoulder_frames_{start_frame}-{end_frame}'] = feedback_msg
                                    
                                    if right_shoulder_y_std > 0.02: # Arbitrary threshold for inconsistency
                                        feedback_msg = f"Right shoulder height inconsistent during draw. Variation: {right_shoulder_y_std:.3f}."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                        video_feedback['improvement_points'].append(f"Draw: {feedback_msg}")
                                    else:
                                        feedback_msg = "Right shoulder height is consistent during draw."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    video_feedback['phase_summaries'][f'draw_right_shoulder_frames_{start_frame}-{end_frame}'] = feedback_msg
                                    video_feedback['phase_summaries'][f'draw_right_shoulder_frames_{start_frame}-{end_frame}'] = feedback_msg

                                # Calculate elbow angles and provide feedback
                                # Assuming get_angle function is available and returns angles for relevant landmarks
                                # This part needs actual angle calculation logic based on your pose data structure
                                # For example, if you have a function like calculate_angle(p1, p2, p3):
                                # right_elbow_angles = [calculate_angle(p[12], p[14], p[16]) for p in phase_pose_data if len(p) > 16]
                                # left_elbow_angles = [calculate_angle(p[11], p[13], p[15]) for p in phase_pose_data if len(p) > 15]

                                # Placeholder for elbow angle feedback
                                # if right_elbow_angles:
                                #     avg_right_elbow_angle = np.mean(right_elbow_angles)
                                #     if avg_right_elbow_angle < 90: # Example threshold
                                #         feedback_msg = f"Right elbow angle too acute: {avg_right_elbow_angle:.2f} degrees. Aim for a more open angle."
                                #         phase_feedback.append(feedback_msg)
                                #         video_feedback['improvement_points'].append(f"Draw: {feedback_msg}")
                                #     video_feedback['phase_summaries'][f'draw_right_elbow_angle_frames_{start_frame}-{end_frame}'] = phase_feedback[-1]

                                # if left_elbow_angles:
                                #     avg_left_elbow_angle = np.mean(left_elbow_angles)
                                #     if avg_left_elbow_angle > 160: # Example threshold for overextension
                                #         feedback_msg = f"Left elbow angle too extended: {avg_left_elbow_angle:.2f} degrees. Maintain a slight bend."
                                #         phase_feedback.append(feedback_msg)
                                #         video_feedback['improvement_points'].append(f"Draw: {feedback_msg}")
                                #     video_feedback['phase_summaries'][f'draw_left_elbow_angle_frames_{start_frame}-{end_frame}'] = phase_feedback[-1]

                            else:
                                phase_feedback.append("No pose data available for draw phase analysis.")
                        elif phase_name == 'setup':
                            if phase_pose_data:
                                # Check left/right hand distance from the bow center (wrist + elbow position)
                                # Using midpoint of shoulders as a proxy for bow center
                                mid_shoulder_x = (phase_pose_data[0][11]['x'] + phase_pose_data[0][12]['x']) / 2
                                mid_shoulder_y = (phase_pose_data[0][11]['y'] + phase_pose_data[0][12]['y']) / 2
                                bow_center_proxy = {'x': mid_shoulder_x, 'y': mid_shoulder_y, 'z': 0} # Z can be ignored for 2D distance

                                # Assuming the bow is held by the left hand (for right-handed archer)
                                left_wrist = phase_pose_data[0][15]
                                left_elbow = phase_pose_data[0][13]

                                dist_left_wrist_bow_center = np.sqrt((left_wrist['x'] - bow_center_proxy['x'])**2 + (left_wrist['y'] - bow_center_proxy['y'])**2)
                                dist_left_elbow_bow_center = np.sqrt((left_elbow['x'] - bow_center_proxy['x'])**2 + (left_elbow['y'] - bow_center_proxy['y'])**2)

                                # Arbitrary thresholds for feedback
                                if dist_left_wrist_bow_center < 0.05: # Too close
                                    feedback_msg = "Bow-side wrist is too close to the body center. Ensure proper extension."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)
                                elif dist_left_wrist_bow_center > 0.2: # Too far
                                    feedback_msg = "Bow-side wrist is too far from the body center. Check your bow arm extension."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)
                                else:
                                    phase_feedback.append("Bow-side wrist position looks good relative to body center.")

                                if dist_left_elbow_bow_center < 0.1: # Too close
                                    phase_feedback.append("Bow-side elbow is too close to the body center. Ensure proper extension.")
                                elif dist_left_elbow_bow_center > 0.3: # Too far
                                    feedback_msg = "Bow-side elbow is too far from the body center. Check your bow arm extension."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)
                                else:
                                    feedback_msg = "Bow-side elbow position looks good relative to body center."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)

                                # Look for arrow-hand consistency by checking wrist/elbow height over multiple frames
                                # Assuming right hand is arrow hand
                                right_wrist_y_coords = [p[16]['y'] for p in phase_pose_data if len(p) > 16]
                                right_elbow_y_coords = [p[14]['y'] for p in phase_pose_data if len(p) > 14]

                                if len(right_wrist_y_coords) > 1:
                                    wrist_y_std = np.std(right_wrist_y_coords)
                                    if wrist_y_std > 0.02: # Arbitrary threshold for inconsistency
                                        feedback_msg = f"Inconsistent arrow-hand wrist height detected. Variation: {wrist_y_std:.3f}."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else:
                                        feedback_msg = "Arrow-hand wrist height is consistent."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                                if len(right_elbow_y_coords) > 1:
                                    elbow_y_std = np.std(right_elbow_y_coords)
                                    if elbow_y_std > 0.02: # Arbitrary threshold for inconsistency
                                        feedback_msg = f"Inconsistent arrow-hand elbow height detected. Variation: {elbow_y_std:.3f}."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else:
                                        feedback_msg = "Arrow-hand elbow height is consistent."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                                # Measure bow grip by distance between bow-side wrist and shoulder
                                # Distance between left wrist (15) and left shoulder (11)
                                if len(phase_pose_data[0]) > 15 and len(phase_pose_data[0]) > 11:
                                    left_wrist_grip = phase_pose_data[0][15]
                                    left_shoulder_grip = phase_pose_data[0][11]
                                    grip_distance = np.sqrt((left_wrist_grip['x'] - left_shoulder_grip['x'])**2 + (left_wrist_grip['y'] - left_shoulder_grip['y'])**2)

                                    if grip_distance < 0.1: # Too close, possibly collapsed grip
                                        feedback_msg = "Bow grip appears too close to the shoulder. Ensure proper bow arm extension."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    elif grip_distance > 0.2: # Too far, possibly overextended
                                        feedback_msg = "Bow grip appears too far from the shoulder. Check your bow arm form."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else:
                                        feedback_msg = "Bow grip distance from shoulder looks good."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)








                                        






                                                                                 # Rate alignment and balance
                                    # This is a summary based on previous checks
                                    balance_issues = [fb for fb in phase_feedback if 'imbalance' in fb or 'stability' in fb or 'balance' in fb]
                                    if not balance_issues:
                                        phase_feedback.append("Overall alignment and balance: Good.")
                                    elif len(balance_issues) == 1:
                                        phase_feedback.append(f"Overall alignment and balance: Fair. {balance_issues[0]}")
                                    else:
                                        phase_feedback.append("Overall alignment and balance: Needs improvement. Multiple issues detected.")
                                else:
                                    phase_feedback.append("No pose data available for stance phase analysis.")
                        elif phase_name == 'anchor':
                            for frame_idx in range(start_frame, end_frame + 1):
                                per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            if phase_pose_data:

                                # 1. Confirm if draw hand lands near a consistent facial anchor point (nose, chin, etc.)
                                # For simplicity, let's use the nose (landmark 0) and right wrist (landmark 16) as anchor points.
                                # We'll check the consistency of the distance between them.
                                nose_wrist_distances = []
                                for pose_data_frame in phase_pose_data:
                                    if len(pose_data_frame) > 16 and len(pose_data_frame) > 0:
                                        nose = pose_data_frame[0]
                                        right_wrist = pose_data_frame[16]
                                        distance = np.sqrt((nose['x'] - right_wrist['x'])**2 + (nose['y'] - right_wrist['y'])**2)
                                        nose_wrist_distances.append(distance)

                                if nose_wrist_distances:
                                    distance_std = np.std(nose_wrist_distances)
                                    # Arbitrary threshold for consistency
                                    if distance_std < 0.01: # Very low variance
                                        feedback_msg = "Draw hand lands consistently near the facial anchor point."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    elif distance_std < 0.03: # Moderate variance
                                        feedback_msg = f"Draw hand shows some variance ({distance_std:.3f}) around the anchor point. Aim for more consistency."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else: # High variance
                                        feedback_msg = f"Draw hand shows significant variance ({distance_std:.3f}) around the anchor point. Focus on a consistent anchor."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                                # Check head stability
                                head_x_coords = [p[0]['x'] for p in phase_pose_data if len(p) > 0]
                                head_y_coords = [p[0]['y'] for p in phase_pose_data if len(p) > 0]

                                if len(head_x_coords) > 1 and len(head_y_coords) > 1:
                                    head_x_std = np.std(head_x_coords)
                                    head_y_std = np.std(head_y_coords)
                                    head_movement_score = (head_x_std + head_y_std) / 2 # Simple average

                                    if head_movement_score < 0.01: # Very stable
                                        feedback_msg = "Head is very stable during the anchor phase."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    elif head_movement_score < 0.03: # Slight drift
                                        feedback_msg = f"Slight head drift detected ({head_movement_score:.3f}). Maintain head stability for better aiming."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else: # Significant movement
                                        feedback_msg = f"Significant head movement detected ({head_movement_score:.3f}). Focus on keeping your head still."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                            # Detect head tilt angle (e.g., angle between eyes and nose)
                            # Using left eye (1), right eye (2), and nose (0)
                            head_tilt_angles = []
                            for pose_data_frame in phase_pose_data:
                                if len(pose_data_frame) > 2:
                                    left_eye = pose_data_frame[1]
                                    right_eye = pose_data_frame[2]
                                    nose = pose_data_frame[0]

                                    # Angle of the line connecting left and right eye relative to horizontal
                                    if (right_eye['x'] - left_eye['x']) != 0:
                                        eye_line_angle_rad = np.arctan2(right_eye['y'] - left_eye['y'], right_eye['x'] - left_eye['x'])
                                        head_tilt_angles.append(np.degrees(eye_line_angle_rad))

                            if head_tilt_angles:
                                avg_head_tilt = np.mean(head_tilt_angles)
                                if abs(avg_head_tilt) > 5: # Arbitrary threshold for tilt
                                    feedback_msg = f"Head tilt detected ({avg_head_tilt:.2f} degrees). Keep your head upright for consistent aiming."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                         per_frame_feedback[frame_idx].append(feedback_msg)
                                else:
                                    feedback_msg = "Head tilt is minimal and consistent."
                                    phase_feedback.append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)
                                    for frame_idx in range(start_frame, end_frame + 1):
                                        per_frame_feedback[frame_idx].append(feedback_msg)

                            else:
                                phase_feedback.append("No pose data available for anchor phase analysis.")

                        elif phase_name == 'setup':
                            for frame_idx in range(start_frame, end_frame + 1):
                                per_frame_feedback[frame_idx].append(f"Phase: {phase_name.title()}")
                            if phase_pose_data:
                                # 1. Bow-side wrist and elbow distance from the body center
                                # Check consistency of bow-side arm extension and position
                                bow_arm_distances = []
                                for pose_data_frame in phase_pose_data:
                                    if len(pose_data_frame) > 11 and len(pose_data_frame) > 13:
                                        left_shoulder = pose_data_frame[11] # Bow side shoulder
                                        left_wrist = pose_data_frame[15] # Bow side wrist
                                        left_elbow = pose_data_frame[13] # Bow side elbow

                                        # Distance from wrist to shoulder (proxy for arm extension)
                                        wrist_shoulder_dist = np.sqrt((left_wrist['x'] - left_shoulder['x'])**2 + (left_wrist['y'] - left_shoulder['y'])**2)
                                        bow_arm_distances.append(wrist_shoulder_dist)

                                if bow_arm_distances:
                                    dist_std = np.std(bow_arm_distances)
                                    if dist_std < 0.02: # Low variance
                                        feedback_msg = "Bow-side arm extension is consistent during setup."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else: # High variance
                                        feedback_msg = f"Bow-side arm extension shows variance ({dist_std:.3f}) during setup. Focus on consistent extension."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                                # 2. Arrow-hand consistency (e.g., consistent grip/placement on string)
                                # This is harder to detect with pose landmarks alone, might need more specific landmarks or object detection.
                                # For now, let's use the right wrist (arrow hand) position relative to the body center.
                                right_wrist_x_coords = []
                                for pose_data_frame in phase_pose_data:
                                    if len(pose_data_frame) > 16:
                                        right_wrist_x_coords.append(pose_data_frame[16]['x'])

                                if right_wrist_x_coords:
                                    wrist_x_std = np.std(right_wrist_x_coords)
                                    if wrist_x_std < 0.015: # Low variance
                                        feedback_msg = "Arrow-hand placement is consistent during setup."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else: # High variance
                                        feedback_msg = f"Arrow-hand placement shows variance ({wrist_x_std:.3f}) during setup. Ensure consistent grip."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                                # 3. Bow grip distance from the shoulder (consistency)
                                # Using left wrist (bow grip) and left shoulder
                                bow_grip_distances = []
                                for pose_data_frame in phase_pose_data:
                                    if len(pose_data_frame) > 11 and len(pose_data_frame) > 15:
                                        left_shoulder = pose_data_frame[11]
                                        left_wrist = pose_data_frame[15]
                                        distance = np.sqrt((left_wrist['x'] - left_shoulder['x'])**2 + (left_wrist['y'] - left_shoulder['y'])**2)
                                        bow_grip_distances.append(distance)

                                if bow_grip_distances:
                                    grip_dist_std = np.std(bow_grip_distances)
                                    if grip_dist_std < 0.02: # Low variance
                                        feedback_msg = "Bow grip distance from shoulder is consistent during setup."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)
                                    else: # High variance
                                        feedback_msg = f"Bow grip distance shows variance ({grip_dist_std:.3f}) during setup. Maintain consistent distance."
                                        phase_feedback.append(feedback_msg)
                                        for frame_idx in range(start_frame, end_frame + 1):
                                            per_frame_feedback[frame_idx].append(feedback_msg)

                            else:
                                phase_feedback.append("No pose data available for setup phase analysis.")

                        # Store phase feedback and scores
                        video_feedback['phase_scores'].update(video_scores)

            # Calculate total performance score
            total_score = 0
            if video_feedback['phase_scores']:
                total_score = np.mean(list(video_feedback['phase_scores'].values()))
            video_feedback['total_performance_score'] = total_score



            all_video_results[video_name] = {
                'phase_data': segmented_phases,
                'per_frame_feedback': per_frame_feedback,
                'phase_summaries': video_feedback.get('phase_summaries', {}),
                'improvement_points': video_feedback.get('improvement_points', []),
                'phase_scores': video_feedback.get('phase_scores', {})
            }

            # Define output filename for individual feedback
            output_filename = os.path.join(feedback_output_dir, f"{video_name}_feedback.json")

            # Save the structured feedback to a JSON file
            with open(output_filename, 'w') as f:
                json.dump(video_feedback, f, indent=4)





    return all_video_results


if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    poses_dir = os.path.join(project_root, 'poses')

    results = process_poses_directory(poses_dir)

    for video, data in results.items():
        print(f"\n--- Feedback for {video} ---")
        # The 'feedback' key might not exist directly in the returned structure
        # You might need to iterate through phase_summaries or improvement_points
        if 'phase_summaries' in data:
            for phase, summary in data['phase_summaries'].items():
                print(f"- {phase}: {summary}")
        if 'improvement_points' in data:
            print(f"\n--- Improvement Points for {video} ---")
            for point in data['improvement_points']:
                print(f"- {point}")
        
        print(f"\n--- Detailed Phase Scores for {video} ---")
        if 'phase_scores' in data:
            for score_type, score_value in data['phase_scores'].items():
                print(f"- {score_type.replace('_', ' ').title()}: {float(score_value):.2f}/10")
        else:
            print("No scores calculated (insufficient data or no relevant angles).")