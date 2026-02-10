import numpy as np
import torch
import logging
import time
import sys

# Ensure tmrl is in path
sys.path.insert(0, 'c:/Users/felix/OneDrive/Documents/Data Mining/tmrl')

# Import your configuration
from tmrl.config.config_objects import ENV_CLS, TRAIN_MODEL, OBS_PREPROCESSOR

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridCheck")

def check_pipeline():
    logger.info("--- STEP 1: Initializing Hybrid Environment ---")
    # 1. Initialize the Environment (uses the class from config_objects.py)
    # This will open the TrackMania window, so make sure the game is running!
    try:
        env = ENV_CLS()
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        logger.error("Make sure TrackMania 2020 is running and the OpenPlanet script is active.")
        return

    logger.info("Environment initialized successfully.")
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Action Space: {env.action_space}")

    # 2. Test Reset
    logger.info("\n--- STEP 2: Testing Reset ---")
    time.sleep(1.0) # Give time to focus window
    obs, info = env.reset()
    
    # Unpack the tuple (Speed, Gear, RPM, Images, Lidar, Act1, Act2)
    # We expect 7 elements because rtgym adds action history
    try:
        if len(obs) == 7:
            speed, gear, rpm, images, lidar, act1, act2 = obs
            logger.info("✅ Observation unpacking successful (7 items found - including action history).")
        else:
             logger.warning(f"⚠️ Unexpected observation length: {len(obs)}")
             speed, gear, rpm, images, lidar = obs[:5] # Try to unpack first 5
             act1, act2 = np.zeros(3), np.zeros(3) # Dummies
             logger.info("✅ Observation unpacking successful (5 items found).")

        logger.info(f"   Speed shape: {speed.shape} | Val: {speed}")
        logger.info(f"   Gear shape:  {gear.shape}  | Val: {gear}")
        logger.info(f"   RPM shape:   {rpm.shape}   | Val: {rpm}")
        logger.info(f"   Img shape:   {images.shape} (History, H, W) or (H, W, C)")
        logger.info(f"   Lidar shape: {lidar.shape}")
        
        # Check if images are grayscale (3 dims: Hist, H, W) or color (4 dims: Hist, H, W, C)
        if len(images.shape) == 3:
            logger.info("   Image Format: Grayscale (Correct for HybridNanoEffNet)")
        elif len(images.shape) == 4 and images.shape[-1] == 1:
            logger.info("   Image Format: Grayscale with channel dim (Acceptable)")
        else:
            logger.warning(f"   Image Format: {images.shape} (Might be Color, check config!)")

    except ValueError as e:
        logger.error(f"❌ Observation unpacking failed! Values: {len(obs)}. Error: {e}")
        return

    # 3. Test Preprocessor
    logger.info("\n--- STEP 3: Testing Preprocessor ---")
    if OBS_PREPROCESSOR:
        # The observation from GenericGymEnv ALREADY contains action history if configured
        # So we pass 'obs' directly if it has 7 items.
        
        if len(obs) == 5:
             # Manually add actions if somehow missing
             dummy_act1 = np.zeros(3)
             dummy_act2 = np.zeros(3)
             full_tuple = (*obs, dummy_act1, dummy_act2)
        else:
             full_tuple = obs
        
        processed_obs = OBS_PREPROCESSOR(full_tuple)
        
        # The preprocessor should return (speed, gear, rpm, norm_images, flat_lidar, act1, act2)
        try:
            p_speed, p_gear, p_rpm, p_imgs, p_lidar, _, _ = processed_obs
            
            logger.info("✅ Preprocessing successful.")
            logger.info(f"   Processed Img Range: [{p_imgs.min():.2f}, {p_imgs.max():.2f}] (Should be 0.0-1.0)")
            logger.info(f"   Processed Lidar Shape: {p_lidar.shape}")
            
            if p_imgs.max() > 1.0:
                 logger.warning("❌ Images are NOT normalized to 0-1 range!")
            
        except ValueError as e:
             logger.error(f"❌ Preprocessor return unpacking failed: {e}")
             return
    else:
        logger.warning("⚠️ No Preprocessor defined in config_objects.py!")

    # 4. Test Model Forward Pass
    logger.info("\n--- STEP 4: Testing Model (VRAM & Dimensions) ---")
    
    # Instantiate the model
    # We need to retrieve the generic spaces to init the model
    try:
        model = TRAIN_MODEL(env.observation_space, env.action_space)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        logger.info(f"✅ Model {type(model).__name__} instantiated on {device}.")
    except Exception as e:
        logger.error(f"❌ Model instantiation failed: {e}")
        return

    # Convert processed obs to Tensor batches (Batch size 1)
    # The model expects: (speed, gear, rpm, images, lidar, act1, act2)
    def to_tensor(x):
        return torch.from_numpy(np.array([x])).float().to(device)

    t_speed = to_tensor(p_speed)
    t_gear = to_tensor(p_gear)
    t_rpm = to_tensor(p_rpm)
    t_imgs = to_tensor(p_imgs)
    t_lidar = to_tensor(p_lidar)
    
    # Use the actions we unpacked or created earlier
    # If they were unpacked from obs, they are available as act1, act2
    # If len(obs)==5, we created dummy_act1, dummy_act2 but didn't assign to act1, act2 in outer scope
    # Let's align them.
    if 'act1' not in locals():
         act1 = np.zeros(3)
    if 'act2' not in locals():
         act2 = np.zeros(3)

    t_act1 = to_tensor(act1)
    t_act2 = to_tensor(act2)

    # Validate image input shape for EffNet
    # EffNet usually expects (Batch, C, H, W) or (Batch, Hist, H, W) mapping to channels
    # The HybridNanoEffNetActor expects 4 dims for images: (Batch, Hist, H, W) which it treats as (Batch, Channel, H, W)
    logger.info(f"   Model Input Image Shape: {t_imgs.shape}")

    # Pack for model
    model_input = (t_speed, t_gear, t_rpm, t_imgs, t_lidar, t_act1, t_act2)

    # Try Actor (Inference)
    try:
        with torch.no_grad():
            action, _ = model.actor(model_input, test=True)
            logger.info(f"✅ Actor Forward Pass successful! Output Action: {action.cpu().numpy()}")
    except Exception as e:
        logger.error(f"❌ Actor Forward Pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try Q-Function (Training)
    try:
        with torch.no_grad():
            # Q-function needs the current action too
            current_act = to_tensor(np.zeros(3)) 
            # Check if using REDQ or SAC
            if hasattr(model, 'qs'): # REDQ
                 q_val = model.qs[0](model_input, current_act)
                 logger.info(f"✅ REDQ Q-Function 1 Forward Pass successful! Q-Value: {q_val.item():.4f}")
            else: # SAC
                 q_val = model.q1(model_input, current_act)
                 logger.info(f"✅ SAC Q-Function 1 Forward Pass successful! Q-Value: {q_val.item():.4f}")
    except Exception as e:
        logger.error(f"❌ Q-Function Forward Pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("\n🎉🎉 CONGRATULATIONS! THE HYBRID PIPELINE IS FULLY FUNCTIONAL 🎉🎉")
    env.close()

if __name__ == "__main__":
    check_pipeline()
