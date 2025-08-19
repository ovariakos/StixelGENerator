# File: create_video.py
import os
import glob
import argparse
import numpy as np
import matplotlib

# NOTE: Use a non-interactive backend for script-based plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def create_point_cloud_video(input_folder, output_file, fps=10):
    """
    Generates a video from a sequence of point cloud CSV files.

    Each frame is a top-down 2D plot, filtered to a <20m range,
    and colored by the Z-level (height).
    """
    csv_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))
    if not csv_files:
        print(f"Error: No CSV files found in '{input_folder}'")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.set_tight_layout(True)
    width = int(fig.get_figwidth() * fig.get_dpi())
    height = int(fig.get_figheight() * fig.get_dpi())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # NOTE: Modified loop to allow updating the description
    with tqdm(csv_files, desc="Starting video creation") as pbar:
        for csv_file in pbar:
            # NOTE: Display the current filename in the progress bar
            pbar.set_description(f"Processing {os.path.basename(csv_file)}")
            try:
                ax.clear()
                points = np.loadtxt(csv_file, delimiter=',', skiprows=1)

                if points.ndim == 1 and points.shape[0] == 0:  # Handle empty files
                    print(f"\nSkipping empty file: {os.path.basename(csv_file)}")
                    continue

                distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
                mask = distances < 20.0
                filtered_points = points[mask]

                # If filtering results in no points, skip this frame
                if filtered_points.shape[0] == 0:
                    continue

                x = filtered_points[:, 0]
                y = filtered_points[:, 1]
                z = filtered_points[:, 2]

                scatter = ax.scatter(x, y, s=1.5, c=z, cmap='coolwarm')

                ax.set_xlim(-20, 20)
                ax.set_ylim(-20, 20)
                ax.set_aspect('equal', adjustable='box')

                ax.plot(0, 0, 'ko', markersize=8, label='Vehicle Origin (0,0)')
                ax.set_xlabel('X - Forward (meters)')
                ax.set_ylabel('Y - Left (meters)')
                ax.set_title(f'Top-Down View Colored by Height\nFrame: {os.path.basename(csv_file)}')

                # --- FIXED SECTION ---
                fig.canvas.draw()
                # NOTE: tostring_rgb() is deprecated. Use buffer_rgba() which is the modern equivalent.
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)
                # NOTE: Convert RGBA (from new method) to BGR (for OpenCV)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                # --- END OF FIX ---

                video_writer.write(img)

            except Exception as e:
                print(f"\nSkipping file {os.path.basename(csv_file)} due to an error: {e}")
                continue

    # Add a color bar to the very last valid frame
    if 'scatter' in locals():
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Z-Level / Height (meters)')
        ax.set_title(f'Top-Down View Colored by Height\nFinal Frame with Colorbar')
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)

    video_writer.release()
    plt.close(fig)
    print(f"\nSuccessfully created video: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a top-down point cloud video from CSV files.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the CSV files.")
    parser.add_argument("--output_file", default="top-view.mp4", help="Name of the output video file.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    args = parser.parse_args()

    create_point_cloud_video(args.input_folder, args.output_file, args.fps)