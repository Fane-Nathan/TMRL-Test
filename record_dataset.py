
import time
import threading
import pickle
import numpy as np
import inputs
import os
import keyboard
from pathlib import Path

from tmrl.networking import RolloutWorker, Buffer
from tmrl.config.config_objects import CONFIG_DICT, POLICY, SAMPLE_COMPRESSOR, OBS_PREPROCESSOR, ENV_CLS
from tmrl.custom.custom_memories import MemoryTMHybrid
import tmrl.config.config_constants as cfg

# Ensure dataset directory exists
DATASET_DIR = Path(cfg.DATASET_PATH)
if not DATASET_DIR.exists():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATASET_DIR / 'data.pkl'

class InputReader:
    def __init__(self):
        self.gas = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.running = True
        self.mode = 'keyboard' # default to keyboard
        
        # Check for gamepad
        try:
            inputs.get_gamepad()
            self.mode = 'gamepad'
            print("Gamepad detected! Using Gamepad.")
            self.thread = threading.Thread(target=self._run_gamepad)
            self.thread.start()
        except Exception:
            print("No Gamepad detected. Using Keyboard (Arrows + Z/S for gas/brake).")
            self.thread = threading.Thread(target=self._run_keyboard)
            self.thread.start()

    def _run_gamepad(self):
        while self.running:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    if event.code == 'ABS_RZ':
                        self.gas = event.state / 255.0
                    elif event.code == 'ABS_Z':
                        self.brake = event.state / 255.0
                    elif event.code == 'ABS_X':
                        self.steer = event.state / 32768.0
            except Exception:
                pass

    def _run_keyboard(self):
        while self.running:
            # Gas
            if keyboard.is_pressed('up') or keyboard.is_pressed('z') or keyboard.is_pressed('w'):
                self.gas = 1.0
            else:
                self.gas = 0.0
            
            # Brake
            if keyboard.is_pressed('down') or keyboard.is_pressed('s'):
                self.brake = 1.0
            else:
                self.brake = 0.0

            # Steer
            if keyboard.is_pressed('right') or keyboard.is_pressed('d'):
                self.steer = 1.0
            elif keyboard.is_pressed('left') or keyboard.is_pressed('q') or keyboard.is_pressed('a'):
                self.steer = -1.0
            else:
                self.steer = 0.0
            
            time.sleep(0.01)

    def get_action(self):
        return np.array([self.gas, self.brake, self.steer], dtype=np.float32)

    def stop(self):
        self.running = False
        self.thread.join()

def record_dataset():
    print("Initializing Input Reader...")
    input_reader = InputReader()
    
    print("Initializing Environment (Worker)...")
    # We use a standalone worker to interface with the game
    worker = RolloutWorker(
        env_cls=ENV_CLS,
        actor_module_cls=POLICY,
        sample_compressor=SAMPLE_COMPRESSOR,
        device='cpu',
        standalone=True,
        obs_preprocessor=OBS_PREPROCESSOR
    )
    
    print(f"\n{'='*50}")
    print(f"RECORDING STARTED")
    print(f"Control the car! Press Ctrl+C to stop and save.")
    print(f"{'='*50}\n")
    
    episodes = 0
    total_steps = 0
    
    try:
        while True:
            # Reset environment
            # We must use collect_samples=True to enable buffering
            obs, info = worker.reset(collect_samples=True)
            done = False
            
            episode_steps = 0
            
            while not done:
                # 1. Get Action from User
                action = input_reader.get_action()
                
                # 2. Step the Environment using the Worker's logic
                # worker.step() normally calls self.act() then env.step()
                # We want to use OUR action, but still use worker.step logic for buffering.
                # However, worker.step() receives 'obs' but calculates 'act' internally.
                # We can hack this by overriding the worker's actor locally or just calling env.step explicitly 
                # and then appending to buffer manually.
                
                # Let's replicate worker.step() logic here for maximum control:
                
                # worker.step(obs, test=False, collect_samples=True) logic:
                # act = self.act(obs) -> REPLACED BY USER INPUT
                # new_obs, rew, terminated, truncated, info = self.env.step(act)
                # ... preprocessing ...
                # self.buffer.append_sample(...)
                
                new_obs, rew, terminated, truncated, info = worker.env.step(action)
                
                if worker.obs_preprocessor is not None:
                    new_obs = worker.obs_preprocessor(new_obs)
                
                # Handling "last step" logic for truncation
                if episode_steps == worker.max_samples_per_episode - 1 and not terminated:
                    truncated = True
                
                # CRC Debug stuff (optional, skipping for now)
                
                # SAMPLE COMPRESSION
                # This is crucial. worker.get_local_buffer_sample must be called.
                if worker.get_local_buffer_sample:
                    sample = worker.get_local_buffer_sample(action, new_obs, rew, terminated, truncated, info)
                else:
                    sample = action, new_obs, rew, terminated, truncated, info
                
                worker.buffer.append_sample(sample)
                
                obs = new_obs
                done = terminated or truncated
                episode_steps += 1
                total_steps += 1
                
                # Optional: Sleep to match roughly real-time if needed? 
                # The environment interface usually handles timing (wait_on_done=True/False)
                # But TMRL is real-time, so it should be fine.
            
            episodes += 1
            print(f"Episode {episodes} finished. Total Samples: {len(worker.buffer)}")

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        input_reader.stop()
        
        if len(worker.buffer) > 0:
            print(f"Processing {len(worker.buffer)} samples...")
            
            # Create a Memory object to format the data
            # MemoryTMHybrid requires nb_steps for init, let's use 1 (doesn't matter for saving)
            memory = MemoryTMHybrid(device='cpu', nb_steps=1, memory_size=1000000, dataset_path=str(DATASET_DIR))
            
            # Append the worker's buffer to memory to format it into columns
            memory.append_buffer(worker.buffer)
            
            print(f"Saving to {DATA_FILE}...")
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(memory.data, f)
            print("Done! Dataset saved.")
        else:
            print("No samples collected.")

if __name__ == "__main__":
    record_dataset()
