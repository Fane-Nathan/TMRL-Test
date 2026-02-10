import cv2
import numpy as np
import math
import sys
import time

# Ensure tmrl is in path
sys.path.append('c:/Users/felix/OneDrive/Documents/Data Mining/tmrl')

from tmrl.custom.tm.utils.window import WindowInterface

def calibrate():
    print("--- LIDAR CALIBRATION TOOL ---")
    print("1. Launch TrackMania 2020.")
    print("2. Ensure the game is visible.")
    print("3. When the image appears, CLICK on the TIP of the CAR NOSE.")
    print("   (The point where you want the LIDAR rays to start)")
    print("------------------------------")
    
    # Give user a moment to switch window if needed
    time.sleep(2.0)

    # 1. Grab Screenshot
    try:
        window_interface = WindowInterface("Trackmania")
        window_interface.move_and_resize()
        img = window_interface.screenshot()[:, :, :3] # BGR
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        print("Make sure Trackmania is running!")
        return
    
    h, w, _ = img.shape
    
    # Current Default Position (44/49 approx 0.89)
    curr_y = 44 * h // 49
    curr_x = w // 2
    
    print(f"Image Resolution: {w}x{h}")
    print(f"Current Lidar Origin: ({curr_x}, {curr_y})")

    # Setup Callback for Click
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\n✅ Clicked at: ({x}, {y})")
            
            # Calculate Ratios
            ratio_h = y / h
            
            print(f"NEW Y Coordinate: {y}")
            print(f"NEW Ratio: {ratio_h:.3f} * h")
            
            # Visual Feedback
            display_img = img.copy()
            
            # Draw Old Lidar (Red)
            cv2.circle(display_img, (curr_x, curr_y), 5, (0, 0, 255), -1)
            cv2.putText(display_img, "OLD", (curr_x + 10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw New Lidar (Green)
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display_img, "NEW (Nose Tip)", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Simulate Rays from New Point
            draw_rays(display_img, (y, x), h, w)
            
            cv2.imshow("Calibrate Lidar", display_img)
            
            print("\nTo apply this fix:")
            print(f"1. Open tmrl/custom/tm/utils/tools.py")
            print(f"2. Find line: self.road_point = (44*h//49, w//2)")
            print(f"3. Change to: self.road_point = ({y}, w//2)")
            print(f"   OR better: self.road_point = (int(h * {ratio_h:.3f}), w//2)")

    def draw_rays(image, origin, h, w):
        # origin is (y, x)
        y, x = origin
        min_dist = 20
        # Draw the fan of rays
        for angle in range(90, 280, 10):
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            # Just draw a short line to show direction
            end_x = int(x + 100 * dy) # dy maps to width (col)
            end_y = int(y + 100 * dx) # dx maps to height (row)
            cv2.line(image, (x, y), (end_x, end_y), (255, 255, 0), 1)

    cv2.imshow("Calibrate Lidar", img)
    cv2.setMouseCallback("Calibrate Lidar", click_event)

    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
