import cv2
import numpy as np
import os

def add_noise(frame, intensity=30):
    """Add random noise to the frame."""
    noise = np.random.randint(-intensity, intensity, frame.shape, dtype='int16')
    noisy_frame = cv2.add(frame.astype('int16'), noise)
    noisy_frame = np.clip(noisy_frame, 0, 255).astype('uint8')
    return noisy_frame

def apply_panning(frame, x_offset=0, y_offset=0):
    """Apply panning effect by translating the frame."""
    rows, cols, _ = frame.shape
    translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    panned_frame = cv2.warpAffine(frame, translation_matrix, (cols, rows))
    return panned_frame

def apply_viewing_angle(frame, tilt=0.1):
    """Apply perspective transformation to simulate an angled view."""
    rows, cols, _ = frame.shape
    src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    dst_points = np.float32([
        [cols * tilt, rows * tilt],
        [cols - cols * tilt, rows * tilt],
        [0, rows],
        [cols, rows]
    ])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    angled_frame = cv2.warpPerspective(frame, perspective_matrix, (cols, rows))
    return angled_frame

def process_video_common(input_path, output_path, frame_processor):
    """Common function to process video frames."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video file {input_path}.")

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame using the provided frame processor
        processed_frame = frame_processor(frame)
        out.write(processed_frame)

    cap.release()
    out.release()

    # Replace the input file with the output file
    os.replace(output_path, input_path)


def apply_noise_to_video(input_path, intensity=30):
    """Apply noise effect to the video in place."""
    temp_path = input_path + ".temp.mp4"

    def processor(frame):
        return add_noise(frame, intensity=intensity)

    process_video_common(input_path, temp_path, processor)

def apply_panning_to_video(input_path, x_pan=5, y_pan=5):
    """Apply panning effect to the video in place."""
    temp_path = input_path + ".temp.mp4"

    def processor(frame):
        return apply_panning(frame, x_offset=x_pan, y_offset=y_pan)

    process_video_common(input_path, temp_path, processor)

def apply_viewing_angle_to_video(input_path, tilt=0.1):
    """Apply viewing angle effect to the video in place."""
    temp_path = input_path + ".temp.mp4"

    def processor(frame):
        return apply_viewing_angle(frame, tilt=tilt)

    process_video_common(input_path, temp_path, processor)
