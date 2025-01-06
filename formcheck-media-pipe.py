import os
import cv2
import mediapipe as mp
import tensorflow as tf

# Ensure TensorFlow uses the GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for use.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("GPU not found. The script will run on the CPU.")

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

mp_drawing = mp.solutions.drawing_utils

def process_video(video_path, output_path):
    """Process a single video to detect poses and save the output."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

def process_folder(input_folder, output_folder):
    """Process all videos in the input folder and save to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions as needed
            video_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"processed_{file_name}")
            print(f"Processing video: {file_name}")
            process_video(video_path, output_path)

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = "reddit_videos"
    output_folder = "processed_videos"

    process_folder(input_folder, output_folder)
