# rtgym interfaces for Trackmania

# standard library imports
import logging
import time
from collections import deque

# third-party imports
import cv2
import gymnasium.spaces as spaces
import numpy as np

# third-party imports
from rtgym import RealTimeGymInterface

# local imports
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import control_gamepad, gamepad_reset, gamepad_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient, save_ghost

# Globals ==============================================================================================================

CHECK_FORWARD = 500  # this allows (and rewards) 50m cuts


# Interface for Trackmania 2020 ========================================================================================

class TM2020Interface(RealTimeGymInterface):
    """
    This is the API needed for the algorithm to control TrackMania 2020
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 gamepad: bool = True,
                 save_replays: bool = False,
                 grayscale: bool = True,
                 resize_to=(64, 64)):
        """
        Base rtgym interface for TrackMania 2020 (Full environment)

        Args:
            img_hist_len: int: history of images that are part of observations
            gamepad: bool: whether to use a virtual gamepad for control
            save_replays: bool: whether to save TrackMania replays on successful episodes
            grayscale: bool: whether to output grayscale images or color images
            resize_to: Tuple[int, int]: resize output images to this (width, height)
        """
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.small_window = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = cfg.REWARD_CONFIG['END_OF_TRACK']
        self.constant_penalty = cfg.REWARD_CONFIG['CONSTANT_PENALTY']

        self.initialized = False
        self._is_eval_episode = False

    def set_eval_mode(self, is_eval: bool):
        """Toggle whether the current episode is an evaluation episode (deterministic)."""
        self._is_eval_episode = is_eval

    def initialize_common(self):
        if self.gamepad:
            import vgamepad as vg
            self.j = vg.VX360Gamepad()
            logging.debug(" virtual joystick in use")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self.reward_function = RewardFunction(reward_data_path=cfg.REWARD_PATH,
                                              nb_obs_forward=cfg.REWARD_CONFIG['CHECK_FORWARD'],
                                              nb_obs_backward=cfg.REWARD_CONFIG['CHECK_BACKWARD'],
                                              nb_zero_rew_before_failure=cfg.REWARD_CONFIG['FAILURE_COUNTDOWN'],
                                              min_nb_steps_before_failure=cfg.REWARD_CONFIG['MIN_STEPS'],
                                              max_dist_from_traj=cfg.REWARD_CONFIG['MAX_STRAY'])
        self.client = TM2020OpenPlanetClient()

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Non-blocking function
        Applies the action given by the RL policy
        If control is None, does nothing (e.g. to record)
        Args:
            control: np.array: [forward,backward,right,left]
        """
        if self.gamepad:
            if control is not None:
                control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append('f')
                if control[1] > 0:
                    actions.append('b')
                if control[2] > 0.5:
                    actions.append('r')
                elif control[2] < -0.5:
                    actions.append('l')
                apply_control(actions)

    def grab_data_and_img(self):
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:  # cv2.resize takes dim as (width, height)
            img = cv2.resize(img, self.resize_to)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1]  # reversed view for numpy RGB convention
        data = self.client.retrieve_data()
        self.img = img  # for render()
        return data, img

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        time.sleep(time_sleep)  # must be long enough for image to be refreshed

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        """
        Non-blocking function
        The agent stays 'paused', waiting in position
        """
        self.send_control(self.get_default_action())
        # Ghost Replays should only capture deterministic eval runs to prevent hard-drive bloat
        if self.save_replays and self._is_eval_episode:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        data, img = self.grab_data_and_img()
        speed = np.array([
            data[0],
        ], dtype='float32')
        gear = np.array([
            data[9],
        ], dtype='float32')
        rpm = np.array([
            data[10],
        ], dtype='float32')
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))
        obs = [speed, gear, rpm, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            terminated = True
            rew += self.finish_reward
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))  # cv2 grayscale images are (h, w)
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))  # cv2 images are (h, w, c)
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """
        must return a Box
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(3, ))

    def get_default_action(self):
        """
        initial action at episode start
        """
        return np.array([0.0, 0.0, 0.0], dtype='float32')


class TM2020InterfaceLidar(TM2020Interface):
    def __init__(self, img_hist_len=1, gamepad=False, save_replays: bool = False):
        super().__init__(img_hist_len, gamepad, save_replays)
        self.window_interface = None
        self.lidar = None

    def grab_lidar_speed_and_data(self):
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([
            data[0],
        ], dtype='float32')
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, imgs))


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):

    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        progress = np.array([0], dtype='float32')
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        progress = np.array([self.reward_function.cur_idx / self.reward_function.datalen], dtype='float32')
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype='float32')
        obs = [speed, progress, imgs]
        end_of_track = bool(data[8])
        info = {}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(low=0.0, high=np.inf, shape=(
            self.img_hist_len,
            19,
        ))  # lidars
        return spaces.Tuple((speed, progress, imgs))



class TM2020InterfaceHybrid(TM2020Interface):
    """
    Hybrid Interface: Returns (Speed, Gear, RPM, Images, LIDAR)
    """
    def __init__(self, img_hist_len=4, gamepad=False, save_replays=False, grayscale=True, resize_to=(64, 64)):
        super().__init__(img_hist_len, gamepad, save_replays, grayscale, resize_to)
        self.lidar = None

    def initialize(self):
        super().initialize()
        # Initialize Lidar tool with a dummy screenshot
        self.lidar = Lidar(self.window_interface.screenshot())

    def grab_data_and_img_and_lidar(self):
        # 1. Capture RAW screenshot (needed for accurate LIDAR)
        img_raw = self.window_interface.screenshot()[:, :, :3] # BGR

        # 2. Compute Lidar from RAW image
        lidar = self.lidar.lidar_20(img=img_raw, show=False)

        # 3. Process Image for CNN (Resize & Grayscale)
        if self.resize_to is not None:
            img = cv2.resize(img_raw, self.resize_to)
        else:
            img = img_raw

        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, ::-1] # RGB

        data = self.client.retrieve_data()
        self.img = img # For render

        return data, img, lidar

    def reset(self, seed=None, options=None):
        self.reset_common()
        self.episode_wall_penalty = 0.0
        data, img, lidar = self.grab_data_and_img_and_lidar()

        speed = np.array([data[0]], dtype='float32')
        gear = np.array([data[9]], dtype='float32')
        rpm = np.array([data[10]], dtype='float32')
        xyz = np.array([data[2], data[3], data[4]], dtype='float32')
        progress_val = 0.0
        progress = np.array([progress_val], dtype='float32')
        self.prev_progress = progress_val

        self.img_hist.clear()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        crash = np.array([0.0], dtype='float32')
        progress_gain = np.array([0.0], dtype='float32')

        # Obs structure matches the HybridNanoEffNet expectation:
        obs = [speed, gear, rpm, imgs, lidar, xyz, progress, crash, progress_gain]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        data, img, lidar = self.grab_data_and_img_and_lidar()

        speed = np.array([data[0]], dtype='float32')
        gear = np.array([data[9]], dtype='float32')
        rpm = np.array([data[10]], dtype='float32')
        xyz = np.array([data[2], data[3], data[4]], dtype='float32')

        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        
        # Calculate progress safely
        datalen = getattr(self.reward_function, 'datalen', 1)
        if datalen == 0: datalen = 1
        curr_prog = self.reward_function.cur_idx / datalen
        progress = np.array([curr_prog], dtype='float32')
        if not hasattr(self, 'prev_progress'):
            self.prev_progress = curr_prog
        progress_gain = np.array([curr_prog - self.prev_progress], dtype='float32')
        self.prev_progress = curr_prog

        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist))

        end_of_track = bool(data[8])
        crash_val = 1.0 if (terminated and not end_of_track) else 0.0
        crash = np.array([crash_val], dtype='float32')

        obs = [speed, gear, rpm, imgs, lidar, xyz, progress, crash, progress_gain]

        info = {}
        
        # --- BALANCED REWARD: Wall-Riding Penalty ---
        # DISABLED: Agent needs to learn to drive forward first.
        # Re-enable once forward driving is stable.
        # min_lidar_dist = np.min(lidar)
        # if min_lidar_dist < 1.0:
        #     wall_penalty = (1.0 - min_lidar_dist) * 0.003
        #     rew -= wall_penalty
        #     info['wall_penalty'] = wall_penalty
        #     if not hasattr(self, 'episode_wall_penalty'):
        #         self.episode_wall_penalty = 0.0
        #     self.episode_wall_penalty += wall_penalty
        # --------------------------------------------

        if end_of_track:
            terminated = True
            rew += self.finish_reward
            if hasattr(self, 'episode_wall_penalty') and self.episode_wall_penalty > 0:
                logging.info(f"🏁 Episode finished. Total LiDAR Wall-Riding Penalty: {-self.episode_wall_penalty:.2f}")
        rew += self.constant_penalty
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1, ))
        gear = spaces.Box(low=0.0, high=6, shape=(1, ))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1, ))

        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH

        if self.grayscale:
            img_space = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img_space = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))

        lidar_space = spaces.Box(low=0.0, high=np.inf, shape=(19,))
        xyz_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        progress_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        crash_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        progress_gain_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        return spaces.Tuple((speed, gear, rpm, img_space, lidar_space, xyz_space, progress_space, crash_space, progress_gain_space))


if __name__ == "__main__":
    pass
