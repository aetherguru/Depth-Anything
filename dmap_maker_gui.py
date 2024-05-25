import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
import os
import time
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import threading


class DepthEstimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Estimation App")

        self.encoder_var = tk.StringVar(value='vitl')
        self.processing = False
        self.stop_processing = False

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Depth Model:").grid(row=0, column=0, pady=5, padx=5)
        encoder_options = ['vits', 'vitb', 'vitl']
        self.encoder_menu = tk.OptionMenu(self.root, self.encoder_var, *encoder_options)
        self.encoder_menu.grid(row=0, column=1, pady=5, padx=5)

        tk.Label(self.root, text="Input Folder:").grid(row=1, column=0, pady=5, padx=5)
        self.input_folder_entry = tk.Entry(self.root, width=50)
        self.input_folder_entry.grid(row=1, column=1, pady=5, padx=5)
        self.input_folder_button = tk.Button(self.root, text="Browse", command=self.browse_input_folder)
        self.input_folder_button.grid(row=1, column=2, pady=5, padx=5)

        tk.Label(self.root, text="Output Folder:").grid(row=2, column=0, pady=5, padx=5)
        self.output_folder_entry = tk.Entry(self.root, width=50)
        self.output_folder_entry.grid(row=2, column=1, pady=5, padx=5)
        self.output_folder_button = tk.Button(self.root, text="Browse", command=self.browse_output_folder)
        self.output_folder_button.grid(row=2, column=2, pady=5, padx=5)

        self.start_button = tk.Button(self.root, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=3, column=0, pady=20, padx=5, sticky=tk.E)

        self.cancel_button = tk.Button(self.root, text="Cancel Processing", command=self.cancel_processing,
                                       state=tk.DISABLED)
        self.cancel_button.grid(row=3, column=1, pady=20, padx=5, sticky=tk.W)

        tk.Label(self.root, text="Overall Progress:").grid(row=4, column=0, pady=5, padx=5)
        self.overall_progress_bar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=400)
        self.overall_progress_bar.grid(row=4, column=1, columnspan=2, pady=5, padx=5)

        self.overall_progress_label = tk.Label(self.root, text="0.00%")
        self.overall_progress_label.grid(row=4, column=3, pady=5, padx=5)

        tk.Label(self.root, text="Current Video Progress:").grid(row=5, column=0, pady=5, padx=5)
        self.current_progress_bar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=400)
        self.current_progress_bar.grid(row=5, column=1, columnspan=2, pady=5, padx=5)

        self.current_progress_label = tk.Label(self.root, text="0.00%")
        self.current_progress_label.grid(row=5, column=3, pady=5, padx=5)

        tk.Label(self.root, text="Processing Rate (FPS):").grid(row=6, column=0, pady=5, padx=5)
        self.fps_label = tk.Label(self.root, text="0.00 FPS")
        self.fps_label.grid(row=6, column=1, pady=5, padx=5)

        tk.Label(self.root, text="Time Elapsed:").grid(row=7, column=0, pady=5, padx=5)
        self.time_elapsed_label = tk.Label(self.root, text="00:00:00")
        self.time_elapsed_label.grid(row=7, column=1, pady=5, padx=5)

        tk.Label(self.root, text="Time Remaining:").grid(row=8, column=0, pady=5, padx=5)
        self.time_remaining_label = tk.Label(self.root, text="00:00:00")
        self.time_remaining_label.grid(row=8, column=1, pady=5, padx=5)

    def browse_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_entry.insert(0, folder_selected)

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_entry.insert(0, folder_selected)

    def start_processing(self):
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()
        encoder = self.encoder_var.get()

        if not input_folder or not output_folder:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.processing = True
        self.stop_processing = False
        self.overall_progress_bar["value"] = 0
        self.current_progress_bar["value"] = 0
        self.overall_progress_label.config(text="0.00%")
        self.current_progress_label.config(text="0.00%")
        self.fps_label.config(text="0.00 FPS")
        self.time_elapsed_label.config(text="00:00:00")
        self.time_remaining_label.config(text="00:00:00")

        thread = threading.Thread(target=self.process_videos, args=(input_folder, output_folder, encoder))
        thread.start()

    def cancel_processing(self):
        if messagebox.askokcancel("Cancel", "Are you sure you want to cancel the processing?"):
            self.stop_processing = True

    def process_videos(self, input_folder, output_folder, encoder):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = initialize_model(device, encoder)
        transform = get_transform()
        os.makedirs(output_folder, exist_ok=True)

        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        total_files = len(video_files)

        for idx, filename in enumerate(video_files):
            if self.stop_processing:
                break
            video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, f"dmap_{filename}")
            self.process_video(video_path, output_video_path, model, transform, device)
            self.update_overall_progress((idx + 1) / total_files * 100)

        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)

        if not self.stop_processing:
            messagebox.showinfo("Completed", "Processing completed successfully.")
        else:
            messagebox.showinfo("Cancelled", "Processing was cancelled.")

    def update_overall_progress(self, value):
        self.overall_progress_bar["value"] = value
        self.overall_progress_label.config(text=f"{value:.2f}%")

    def update_current_progress(self, value):
        self.current_progress_bar["value"] = value
        self.current_progress_label.config(text=f"{value:.2f}%")

    def update_fps(self, fps):
        self.fps_label.config(text=f"{fps:.2f} FPS")

    def update_time_labels(self, start_time, idx, total_frames):
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (idx + 1)) * (total_frames - (idx + 1))

        self.time_elapsed_label.config(text=self.format_time(elapsed_time))
        self.time_remaining_label.config(text=self.format_time(remaining_time))

    @staticmethod
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}"

    def process_video(self, video_path, output_video_path, model, transform, device):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        out_video = initialize_video_writer(cap, output_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_time = time.time()
        try:
            for idx in range(total_frames):
                if self.stop_processing:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, model, transform, device)
                out_video.write(processed_frame)
                self.update_current_progress((idx + 1) / total_frames * 100)
                elapsed_time = time.time() - start_time
                fps = (idx + 1) / elapsed_time
                self.update_fps(fps)
                self.update_time_labels(start_time, idx, total_frames)
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
    return cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)


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
        Resize(width=532, height=532, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method='lower_bound', image_interpolation_method=cv2.INTER_LANCZOS4),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def process_frame(frame, model, transform, device):
    image = transform_image(frame, transform, device)
    depth = estimate_depth(image, model)
    depth_grayscale = visualize_depth(depth)
    depth_grayscale_smoothed = apply_smoothing(depth_grayscale)
    return resize_depth_to_frame(depth_grayscale_smoothed, frame)


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
    return depth_rescaled.astype(np.uint8)


def apply_smoothing(depth_grayscale):
    return cv2.GaussianBlur(depth_grayscale, (5, 5), 0)


def resize_depth_to_frame(depth_grayscale, frame):
    return cv2.resize(depth_grayscale, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthEstimationApp(root)
    root.mainloop()
