import time
import logging
import torch
import numpy as np
import cv2
import math
from collections import deque
import sys

# Ensure tmrl is in path
sys.path.append('c:/Users/felix/OneDrive/Documents/Data Mining/tmrl')

# TMRL Imports
from tmrl.config.config_objects import ENV_CLS, TRAIN_MODEL, OBS_PREPROCESSOR

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Visualizer")

def draw_lidar_radar(lidar_data, width=300, height=300):
    """
    Draws the 19 LIDAR beams as rays originating from the bottom center.
    lidar_data: Array of 19 float distances.
    """
    # Create a black background
    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define origin (bottom center)
    origin = (width // 2, height - 20)
    
    # 19 beams spread over 180 degrees (or slightly less, typically ~150 deg for TMRL)
    # TMRL usually spans -90 to +90 degrees or similar. We'll map 19 beams to a semi-circle.
    num_beams = len(lidar_data)
    angle_step = 180 / (num_beams + 1)
    
    # Max distance for normalization (Assumed max valid range ~1000m or tuned for visuals)
    # If the car is looking at the sky, values can be huge. We clamp for visualization.
    max_display_dist = 200.0 
    
    for i, dist in enumerate(lidar_data):
        # Normalize length
        norm_dist = min(dist, max_display_dist) / max_display_dist
        line_len = norm_dist * (height - 40)
        
        # Calculate Angle (0 degrees is Right, 180 is Left in OpenCV coords usually)
        # But we want 0 to be left and 180 right for the loop, or centered.
        # Let's say beam 0 is Left, beam 18 is Right.
        angle_deg = 180 - (i + 1) * angle_step
        angle_rad = math.radians(angle_deg)
        
        # Calculate End Point
        end_x = int(origin[0] + line_len * math.cos(angle_rad))
        end_y = int(origin[1] - line_len * math.sin(angle_rad))
        
        # Color based on distance (Red = Close, Green = Far)
        color = (0, int(255 * norm_dist), int(255 * (1 - norm_dist)))
        
        # Draw Line
        cv2.line(radar_img, origin, (end_x, end_y), color, 2)
        # Draw dot at end
        cv2.circle(radar_img, (end_x, end_y), 3, color, -1)

    # Add Text Stats
    cv2.putText(radar_img, f"Min Dist: {np.min(lidar_data):.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return radar_img

def show_dashboard(images, lidar_data, speed, action, scale=3.0):
    """
    Combines Camera + Lidar + Stats into one window
    """
    # 1. Process Images (Stack Vertical)
    imshape = images.shape
    if len(imshape) == 3: # (History, H, W)
        nb, h, w = imshape
        concat_cam = images.reshape((nb*h, w))
        # Resize Camera
        cam_h = int(concat_cam.shape[0] * scale)
        cam_w = int(concat_cam.shape[1] * scale)
        cam_view = cv2.resize(concat_cam, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        # Convert to BGR for concatenation
        cam_view = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2BGR)
    elif len(imshape) == 4: # (History, H, W, C)
        nb, h, w, c = imshape
        concat_cam = images.reshape((nb*h, w, c))
        # Resize
        cam_h = int(concat_cam.shape[0] * scale)
        cam_w = int(concat_cam.shape[1] * scale)
        cam_view = cv2.resize(concat_cam, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
    else:
        # Fallback
        cam_view = np.zeros((300, 300, 3))

    # 2. Draw Radar
    radar_view = draw_lidar_radar(lidar_data, width=cam_view.shape[1], height=cam_view.shape[0])
    
    # 3. Combine Side-by-Side
    dashboard = np.hstack((cam_view, radar_view))
    
    # 4. Add Telemetry Overlay
    # Action Bar (Steering)
    steer = action[2]
    bar_width = 200
    center_x = dashboard.shape[1] // 2
    cv2.rectangle(dashboard, (center_x - bar_width//2, dashboard.shape[0]-40), (center_x + bar_width//2, dashboard.shape[0]-20), (50, 50, 50), -1)
    # Indicator
    ind_x = int(center_x + (steer * (bar_width//2)))
    cv2.circle(dashboard, (ind_x, dashboard.shape[0]-30), 8, (0, 0, 255), -1)
    
    cv2.putText(dashboard, f"Speed: {speed[0]*1000:.0f} km/h", (20, dashboard.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Hybrid Agent Dashboard", dashboard)
    cv2.waitKey(1)

def run_interactive():
    logger.info("--- Initializing Hybrid Environment ---")
    env = ENV_CLS()
    
    logger.info("--- Loading Model (Untrained Initial Weights) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TRAIN_MODEL(env.observation_space, env.action_space)
    model.to(device)
    model.eval() 
    
    logger.info(f"Model loaded on {device}. Ready to drive!")
    
    # --- SIMULATION LOOP ---
    try:
        while True:
            obs, info = env.reset()
            act_buf = deque(maxlen=2)
            act_buf.append(np.zeros(3))
            act_buf.append(np.zeros(3))
            
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # 1. Unpack (Ensure this matches your Interface!)
                # Handle rtgym's automatic action history (7 items vs 5 items)
                if len(obs) == 7:
                    speed, gear, rpm, images, lidar, act1, act2 = obs
                    full_obs = obs
                    # Update local view of action buffer for visualization if needed, 
                    # but we'll trust the env buffer for the model
                else:
                    speed, gear, rpm, images, lidar = obs
                    full_obs = (speed, gear, rpm, images, lidar, act_buf[0], act_buf[1])
                
                # 2. Preprocess
                if OBS_PREPROCESSOR:
                    processed_obs = OBS_PREPROCESSOR(full_obs)
                else:
                    processed_obs = full_obs

                # Unpack processed (assumes 7 items)
                # OBS_PREPROCESSOR returns (speed, gear, rpm, images, lidar, act1, act2)
                p_speed, p_gear, p_rpm, p_imgs, p_lidar, p_act1, p_act2 = processed_obs
                
                def to_tensor(x):
                    return torch.from_numpy(np.array([x])).float().to(device)

                t_inputs = (to_tensor(p_speed), to_tensor(p_gear), to_tensor(p_rpm), 
                            to_tensor(p_imgs), to_tensor(p_lidar), 
                            to_tensor(p_act1), to_tensor(p_act2))

                # 3. Inference
                with torch.no_grad():
                    if hasattr(model, 'act'):
                        action = model.act(t_inputs, test=True)
                    else:
                        action, _ = model.actor(t_inputs, test=True)
                        action = action.cpu().numpy()[0]

                # 4. Visualization (Raw Lidar data + Images)
                # We use 'lidar' (raw from env) instead of 'p_lidar' for better visualization logic
                # Ensure lidar handles flattening if preprocessor didn't flatten it for the raw view?
                # Actually 'lidar' is raw from unpack, so it's fine.
                show_dashboard(images, lidar, speed, action, scale=4.0)

                # 5. Step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update manual buffer (fallback)
                act_buf.append(action)

            logger.info("Episode finished.")
            
    except KeyboardInterrupt:
        logger.info("\nStopping...")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_interactive()
