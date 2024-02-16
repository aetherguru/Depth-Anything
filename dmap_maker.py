import argparse
import cv2
import numpy as np
import torch
import os
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch process videos for depth estimation with matched output resolution.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Encoder for the depth model.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the input folder containing video files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder for processed videos.')
    return parser.parse_args()

def initialize_model(device, encoder):
    model_path = f'LiheYoung/depth_anything_{encoder}14'
    model = DepthAnything.from_pretrained(model_path).to(device)
    model.eval()
    print_model_parameters(model)
    return model

def print_model_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')

def get_transform():
    return Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def batch_process_videos(input_folder, output_folder, model, transform, device):
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):  # Add other video formats as needed
            continue
        video_path = os.path.join(input_folder, filename)
        output_video_path = os.path.join(output_folder, f"processed_{filename}")
        print(f"Processing {video_path}...")
        process_video(video_path, output_video_path, model, transform, device)
        print(f"Finished processing. Output saved to {output_video_path}")

def process_video(video_path, output_video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    out_video = initialize_video_writer(cap, output_video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, model, transform, device, cap)
            out_video.write(processed_frame)
            cv2.imshow('Depth Anywhere', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

def initialize_video_writer(cap, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def process_frame(frame, model, transform, device, cap):
    image = transform_image(frame, transform, device)
    depth = estimate_depth(image, model)
    depth_color = visualize_depth(depth)
    # Resize depth map to match the frame's resolution
    depth_color_resized = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    return depth_color_resized

def transform_image(frame, transform, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    transformed = transform({'image': frame_rgb})['image']
    return torch.from_numpy(transformed).unsqueeze(0).to(device)

def estimate_depth(image, model):
    with torch.no_grad():
        depth = model(image)
    return depth

def visualize_depth(depth):
    depth_rescaled = depth.squeeze().cpu().numpy()
    depth_rescaled = (depth_rescaled - depth_rescaled.min()) / (depth_rescaled.max() - depth_rescaled.min()) * 255.0
    depth_color = cv2.applyColorMap(depth_rescaled.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return depth_color

def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = initialize_model(device, args.encoder)
    transform = get_transform()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    batch_process_videos(args.input_folder, args.output_folder, model, transform, device)

if __name__ == "__main__":
    main()
