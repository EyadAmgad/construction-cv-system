import cv2
import os

def split_video(input_path, num_parts=9, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frames_per_part = total_frames // num_parts
    print(f"Total frames: {total_frames}, FPS: {fps}, Parts: {num_parts}, Frames per part: {frames_per_part}")

    for part in range(1, num_parts + 1):
        output_path = os.path.join(output_dir, f'train_part_{part}.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_to_write = frames_per_part if part < num_parts else (total_frames - (part - 1) * frames_per_part)
        print(f"Writing {output_path} ({frames_to_write} frames)...")
        
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Saved {output_path}")

    cap.release()
    print("Video splitting complete!")

if __name__ == "__main__":
    split_video('train.mp4', num_parts=9)
