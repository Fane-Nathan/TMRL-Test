# third-party imports
import numpy as np
import logging


# OBSERVATION PREPROCESSING ==================================


def obs_preprocessor_tm_act_in_obs(obs):
    """
    Preprocessor for TM2020 full environment with grayscale images
    """
    grayscale_images = obs[3]
    grayscale_images = grayscale_images.astype(np.float32) / 256.0
    obs = (obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10000.0, grayscale_images, *obs[4:])  # >= 1 action
    return obs


def obs_preprocessor_tm_lidar_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), *obs[2:])  # >= 1  action
    return obs


def obs_preprocessor_tm_lidar_progress_act_in_obs(obs):
    """
    Preprocessor for the LIDAR environment, flattening LIDARs
    """
    obs = (obs[0], obs[1], np.ndarray.flatten(obs[2]), *obs[3:])  # >= 1  action
    return obs


# SAMPLE PREPROCESSING =======================================
# these can be called when sampling from the replay memory, on the whole sample
# this is useful in particular for data augmentation
# be careful whatever you do here is consistent, because consistency after this will NOT be checked by CRC



def sample_preprocessor_tm_lidar_act_in_obs(last_obs, act, rew, new_obs, terminated, truncated):
    return last_obs, act, rew, new_obs, terminated, truncated


def obs_preprocessor_tm_hybrid(obs):
    """
    Preprocessor for TM2020 Hybrid (CNN + Lidar)
    Input: (speed, gear, rpm, images, lidar, act1, act2)
    """
    speed, gear, rpm, images, lidar, act1, act2 = obs

    # Normalize images (0-255 -> 0.0-1.0) for the CNN
    images = images.astype(np.float32) / 255.0

    # Normalize float inputs
    speed = speed / 1000.0
    gear = gear / 10.0
    rpm = rpm / 10000.0

    # Ensure lidar is flat and normalized
    lidar = np.ndarray.flatten(lidar).astype(np.float32) / 300.0

    return (speed, gear, rpm, images, lidar, act1, act2)
