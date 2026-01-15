import torch
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor
import os
import glob
import shutil

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")

VIDEO_INPUT = input("Please enter the video folder path or mp4 file path: ")
TEXT_EXPRESSION = input("Please enter the text expression: ")
OUTPUT_FOLDER = "output"

print(f"Video Source: {VIDEO_INPUT} | Prompt: {TEXT_EXPRESSION}")

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360
        saturation = 0.9 + (i % 2) * 0.1
        value = 0.95 + (i % 2) * 0.05
        h, c = hue / 60.0, value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        if 0 <= h < 1: r, g, b = c, x, 0
        elif 1 <= h < 2: r, g, b = x, c, 0
        elif 2 <= h < 3: r, g, b = 0, c, x
        elif 3 <= h < 4: r, g, b = 0, x, c
        elif 4 <= h < 5: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
        colors.append([int(r), int(g), int(b)])
    return colors

def visualize_frame(frame, outputs, id_to_color):
    img = frame.copy()
    if len(outputs['out_obj_ids']) == 0: return img
    
    for obj_idx, obj_id in enumerate(outputs['out_obj_ids']):
        mask = outputs['out_binary_masks'][obj_idx]
        if torch.is_tensor(mask): mask = mask.cpu().numpy()
        if mask.ndim == 3: mask = mask[0]
        if not np.any(mask > 0): continue

        color_np = id_to_color[obj_id] 
        img[mask > 0] = img[mask > 0] * 0.6 + color_np * 0.4
        
        mask_uint8 = mask.astype(np.uint8)
        M = cv2.moments(mask_uint8)
        
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            label = f"{obj_id}"
            
            font_scale = 0.6
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            text_x = centroid_x - text_width // 2
            text_y = centroid_y + text_height // 2
            
            color_rgb_tuple = (int(color_np[0]), int(color_np[1]), int(color_np[2]))
            
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (0,0,0), thickness + 2)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color_rgb_tuple, thickness)

    return img

def main():
    predictor = build_sam3_video_predictor(gpus_to_use=[0], checkpoint_path='sam3_ckpt/sam3.pt')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    TEMP_VIDEO_FOLDER = OUTPUT_FOLDER + "_temp_frames"
    if os.path.exists(TEMP_VIDEO_FOLDER): shutil.rmtree(TEMP_VIDEO_FOLDER)
    os.makedirs(TEMP_VIDEO_FOLDER, exist_ok=True)

    if os.path.isfile(VIDEO_INPUT):
        cap = cv2.VideoCapture(VIDEO_INPUT)
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(os.path.join(TEMP_VIDEO_FOLDER, f"{saved_count:05d}.jpg"), frame)
            saved_count += 1
        cap.release()
    else:
        frame_paths = sorted(glob.glob(os.path.join(VIDEO_INPUT, "*.jpg")) + glob.glob(os.path.join(VIDEO_INPUT, "*.png")))
        for new_idx, frame_path in enumerate(frame_paths):
            ext = os.path.splitext(frame_path)[1]
            os.symlink(os.path.abspath(frame_path), os.path.join(TEMP_VIDEO_FOLDER, f"{new_idx:05d}{ext}"))

    video_frames = []
    for fp in sorted(glob.glob(os.path.join(TEMP_VIDEO_FOLDER, "*"))):
        video_frames.append(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB))

    response = predictor.handle_request(dict(type="start_session", resource_path=TEMP_VIDEO_FOLDER))
    session_id = response["session_id"]
    predictor.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=0, text=TEXT_EXPRESSION))

    MAX_COLOR_POOL = 200
    color_palette = generate_distinct_colors(MAX_COLOR_POOL)

    processed_results = []

    for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]
        
        current_id_to_color = {}
        for obj_id in outputs['out_obj_ids']:
            color = color_palette[obj_id % MAX_COLOR_POOL]
            current_id_to_color[obj_id] = np.array(color, dtype=np.uint8)

        vis_frame = visualize_frame(video_frames[frame_idx], outputs, current_id_to_color)

        save_path = os.path.join(OUTPUT_FOLDER, f"{frame_idx:05d}.jpg")
        
        processed_results.append((save_path, vis_frame))

    for save_path, frame_data in processed_results:
        Image.fromarray(frame_data.astype(np.uint8)).save(save_path)

    shutil.rmtree(TEMP_VIDEO_FOLDER)

if __name__ == "__main__":
    main()