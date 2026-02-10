
import time
import threading
import pickle
import numpy as np
import inputs
from tmrl.networking import RolloutWorker
from tmrl.config.config_objects import CONFIG_DICT, POLICY, SAMPLE_COMPRESSOR, OBS_PREPROCESSOR
import tmrl.config.config_constants as cfg

class GamepadReader:
    def __init__(self):
        self.gas = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
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

    def get_action(self):
        return np.array([self.gas, self.brake, self.steer], dtype=np.float32)

    def stop(self):
        self.running = False
        self.thread.join()

def record_dataset():
    print("Initializing Gamepad...")
    gamepad = GamepadReader()
    
    print("Initializing Environment...")
    worker = RolloutWorker(
        env_cls=CONFIG_DICT['env'],
        actor_module_cls=POLICY,
        sample_compressor=SAMPLE_COMPRESSOR,
        device='cpu',
        standalone=True,
        obs_preprocessor=OBS_PREPROCESSOR
    )
    
    print("Starting Recording... Press Ctrl+C to stop and save.")
    print("Drive smoothly! converting raw inputs to actions...")
    
    episodes = 0
    try:
        while True:
            worker.reset(collect_samples=True)
            done = False
            while not done:
                action = gamepad.get_action()
                # Apply action (the env might need to be stepped manually or via act)
                # RolloutWorker.step calls self.act() then self.env.step()
                # We want to BYPASS the policy.
                
                # Manually step the environment
                # Note: RolloutWorker.step() combines act() and env.step(). 
                # We need to manually do what worker.step() does but with OUR action.
                
                # Get last obs (we need to track it)
                # Actually, worker has a run_episode() loop. We should implement our own loop using worker methods.
                
                # ... check worker implementation ...
                pass
            episodes += 1
            print(f"Episode {episodes} complete. Buffer size: {len(worker.buffer)}")
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        gamepad.stop()
        print(f"Saving dataset with {len(worker.buffer)} samples to {cfg.DATASET_PATH}...")
        # We need to format the buffer data correctly for Memory
        # The worker buffer is a list of tuples.
        # We can just pickle the list of tuples? NO. Memory expects a list of columns.
        
        # Let's use a dummy Memory object to format it?
        # memory = MemoryTMHybrid(nb_steps=1)
        # memory.append_buffer(worker.buffer)
        # pickle.dump(memory.data, open(cfg.DATASET_PATH, 'wb'))
        pass

if __name__ == "__main__":
    record_dataset()
