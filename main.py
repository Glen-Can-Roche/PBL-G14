from ultralytics import YOLO
import cv2
import numpy as np
import torch

# --- Configuration ---
MODEL_PATH = "best.pt"  # Trained YOLO model
VIDEO_PATH = "input_videos/3936991-hd_1920_1080_30fps.mp4"  # Input video path
OUTPUT_PATH = "output_video.mp4"                   # Output video path
THRESHOLD = 20                                      # People count threshold for alert
CLUSTER_THRESHOLD = 15                               # Cluster size threshold for alert
CLUSTER_DISTANCE_THRESH = 50                        # Distance threshold for clustering people

# Heatmap settings
HEATMAP_ALPHA = 0.5         # transparency of heatmap overlay (0..1)
HEATMAP_BLUR = 51           # Gaussian blur kernel size (odd)
HEATMAP_THRESH = 0.1       # normalized density threshold to mark dense clusters (0..1) reduce for larger heatmap areas
DRAW_CONTOURS = True        # draw contours around dense areas

# Box/label settings
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
LABEL_BG = (0, 128, 0)
LABEL_FG = (255, 255, 255)


def draw_label(img, text, left_top, bg_color=(0, 0, 0), fg_color=(255, 255, 255)):
    """Draws a text label with a filled background rectangle."""
    x, y = left_top
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Adjust y to ensure the label is always within the frame top
    y = max(y, th + baseline + 6)
    # Draw the background rectangle
    cv2.rectangle(img, (x, y - th - baseline - 6), (x + tw + 6, y), bg_color, -1)
    # Draw the text
    cv2.putText(img, text, (x + 3, y - 6), font, scale, fg_color, thickness, cv2.LINE_AA)


# --- MODIFICATION START ---
# Updated function to create a more dynamic heatmap
def build_density(h: int, w: int, centers: np.ndarray, distance_threshold: float) -> np.ndarray:
    """
    Builds a density map from center points, with heat intensity adjusted by local proximity.
    
    Args:
        h (int): Height of the frame.
        w (int): Width of the frame.
        centers (np.ndarray): N x 2 array of (cx, cy) coordinates of detected people.
        distance_threshold (float): Distance to consider for local density.
                                    People within this distance contribute more to each other's "heat".

    Returns:
        np.ndarray: Normalized density map (0.0 - 1.0).
    """
    heat = np.zeros((h, w), dtype=np.float32)
    
    if len(centers) == 0:
        return heat  # Return empty heat map if no centers

    # Calculate local density factor for each person
    # This loop is O(N^2), so for very large crowds, this can become slow.
    # For typical crowd scenes, it should be acceptable.
    density_factors = np.ones(len(centers), dtype=np.float32)  # Each person contributes at least 1.0
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_threshold:
                # If two people are close, increase their mutual "heat contribution"
                # This makes the initial "dot" brighter for people in clusters.
                density_factors[i] += 1.0  # Person i's heat increases because of j
                density_factors[j] += 1.0  # Person j's heat increases because of i

    # Add "heat" for each center point, weighted by the calculated density factor
    for i, (cx, cy) in enumerate(centers):
        if 0 <= cx < w and 0 <= cy < h:
            # Use the density_factor here to make the dot brighter if more people are nearby
            heat[cy, cx] += density_factors[i]
    
    # Apply Gaussian blur to create a smooth density map
    k = max(3, int(HEATMAP_BLUR))
    if k % 2 == 0:
        k += 1  # Kernel size must be odd
    density = cv2.GaussianBlur(heat, (k, k), 0)
    
    # Normalize the density map
    if density.max() > 0:
        density /= (density.max() + 1e-6)  # Avoid division by zero
    return density
# --- MODIFICATION END ---


def cluster_people(boxes, distance_threshold):
    """Clusters people based on the distance between their bounding boxes."""
    if len(boxes) == 0:
        return []

    centers = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes])
    clusters = []
    visited = [False] * len(boxes)

    for i in range(len(boxes)):
        if not visited[i]:
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                current_index = queue.pop(0)
                for j in range(len(boxes)):
                    if not visited[j]:
                        dist = np.linalg.norm(centers[current_index] - centers[j])
                        if dist < distance_threshold:
                            visited[j] = True
                            cluster.append(j)
                            queue.append(j)
            clusters.append(cluster)
    
    return [[boxes[i] for i in cluster] for cluster in clusters]

def main():
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video: {width}x{height} @ {fps} FPS")

    # --- NEW CODE: DYNAMIC IMGZ ---
    # Get the longer dimension of the video
    long_side = max(width, height)
    
    # Clamp the max resolution to 1280 to prevent OOM errors and keep speed reasonable
    # You can increase 1280 to 1920 if you have a very powerful GPU
    max_allowed_imgsz = 1920
    min_allowed_imgsz = 640
    
    if long_side > max_allowed_imgsz:
        imgsz_to_use = max_allowed_imgsz
    elif long_side < min_allowed_imgsz:
        imgsz_to_use = min_allowed_imgsz
    else:
        imgsz_to_use = long_side
        
    # Round to the nearest multiple of 32 (which is required by YOLO)
    dynamic_imgsz = int(np.ceil(imgsz_to_use / 32) * 32)
    
    print(f"Original Resolution: {width}x{height}. Using dynamic 'imgsz={dynamic_imgsz}' for detection.")
    # --- END NEW CODE ---

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Create a resizable window that maintains the aspect ratio for display
    window_name = "Crowd Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)



    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Run YOLO detection
        # --- MODIFIED LINE ---
        # Use the dynamic_imgsz variable instead of a fixed number
        results = model(frame, imgsz=dynamic_imgsz, device=device)[0]
        # --- END MODIFICATION ---
        
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            clss = np.zeros((0,), dtype=int)
            confs = np.zeros((0,), dtype=np.float32)

        # Filter to person class (COCO id 0)
        person_mask = (clss == 0)
        p_xyxy = xyxy[person_mask]
        p_confs = confs[person_mask]

        # Count people
        people_count = p_xyxy.shape[0]

        # Draw boxes and labels
        for (x1, y1, x2, y2), conf in zip(p_xyxy.astype(int), p_confs):
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            draw_label(frame, f"person {conf*100:.1f}%", (x1, y1), bg_color=LABEL_BG, fg_color=LABEL_FG)

        # Cluster people
        clusters = cluster_people(p_xyxy, CLUSTER_DISTANCE_THRESH)

        # Cluster-based alert
        alert_triggered = False
        for cluster in clusters:
            cluster_size = len(cluster)
            if cluster_size >= CLUSTER_THRESHOLD:
                alert_triggered = True
                # Draw a rectangle around the cluster
                x_min = int(min(box[0] for box in cluster))
                y_min = int(min(box[1] for box in cluster))
                x_max = int(max(box[2] for box in cluster))
                y_max = int(max(box[3] for box in cluster))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                draw_label(frame, f"CROWD ALERT! Count: {cluster_size}", (x_min, y_min - 10), bg_color=(0, 0, 255), fg_color=(255, 255, 255))



        # People count overlay
        draw_label(frame, f"People Count: {people_count}", (50, 50), bg_color=(0, 0, 0), fg_color=(255, 255, 255))

        # --- HEATMAP GENERATION ---
        # 1. Get center points of all detected people
        centers = (((p_xyxy[:, 0:2] + p_xyxy[:, 2:4]) / 2.0)).astype(int) if people_count > 0 else np.zeros((0, 2), dtype=int)
        
        # --- MODIFICATION START ---
        # 2. Build the density map
        # Pass the centers and distance threshold to the updated function
        density = build_density(height, width, centers, CLUSTER_DISTANCE_THRESH)
        # --- MODIFICATION END ---
        
        # 3. Apply a color map
        heat_color = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 4. Blend the heatmap with the original frame
        overlay = cv2.addWeighted(heat_color, HEATMAP_ALPHA, frame, 1 - HEATMAP_ALPHA, 0)
        # --- END HEATMAP ---

        # Highlight dense clusters
        mask = (density >= HEATMAP_THRESH).astype(np.uint8) * 255
        if DRAW_CONTOURS and mask.any():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Show and write
        cv2.imshow(window_name, overlay)
        out.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, exiting.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

