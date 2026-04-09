# standard library imports
import os
import pickle

# third-party imports
import numpy as np
import logging


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """
    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=10,
                 nb_obs_backward=10,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(3.5 * 20),
                 max_dist_from_traj=60.0):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
        """
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.cur_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)

        # self.traj = []

    def compute_reward(self, pos):
        """
        Computes the current reward given the position pos
        Args:
            pos: the current position
        Returns:
            float, bool: the reward and the terminated signal
        """
        # self.traj.append(pos)

        terminated = False
        self.step_counter += 1  # step counter to enable failure counter
        min_dist = np.inf  # smallest distance found so far in the trajectory to the target pos
        index = self.cur_idx  # cur_idx is where we were last step in the trajectory
        temp = self.nb_obs_forward  # counter used to find cuts
        best_index = 0  # index best matching the target pos

        while True:
            dist = np.linalg.norm(pos - self.data[index])  # distance of the current index to target pos
            if dist <= min_dist:  # if dist is smaller than our minimum found distance so far,
                min_dist = dist  # then we found a new best distance,
                best_index = index  # and a new best index
                temp = self.nb_obs_forward  # we will have to check this number of positions to find a possible cut
            index += 1  # now we will evaluate the next index in the trajectory
            temp -= 1  # so we can decrease the counter for cuts
            # stop condition
            if index >= self.datalen or temp <= 0:  # if trajectory complete or cuts counter depleted
                # We check that we are not too far from the demo trajectory:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx  # if so, consider we didn't move

                break  # we found the best index and can break the while loop

        # The reward is proportional to the number of NEW passed indexes (track distance)
        # We only reward strictly forward progress to prevent back-and-forth farming
        progress = 0.0
        if best_index > self.cur_idx:
            progress = float(best_index - self.cur_idx)
            reward = progress / 100.0
            self.cur_idx = best_index # update our maximum reached index
        else:
            # We either idled or went backward. 
            # We issue exactly 0.0 reward (no negative penalty, so the AI doesn't fear braking)
            reward = 0.0
            best_index = self.cur_idx # prevent cur_idx from moving backward
            
        # --- PACE CAR / ANTI-CRAWL MECHANIC ---
        # The agent enjoys full immunity during the starting sequence
        if self.step_counter > self.min_nb_steps_before_failure:
            # The Pace Car builds pressure every single frame (+1 point)
            self.failure_counter += 1
            
            # Driving fast relieves pressure (e.g. at least 1 index per 2 frames is breakeven speed)
            if progress > 0.0:
                self.failure_counter = max(0, self.failure_counter - int(progress * 2))
                
            # If the agent crawls or idles too long, the pressure bar fills up and terminates the run.
            # We multiply by 4 to give the agent a massive 40-frame (2 second) buffer for taking corners.
            if self.failure_counter > (self.nb_zero_rew_before_failure * 4):
                terminated = True

        return reward, terminated

    def reset(self):
        """
        Resets the reward function for a new episode.
        """
        # from pathlib import Path
        # import pickle as pkl
        # path_traj = Path.home() / 'TmrlData' / 'reward' / 'traj.pkl'
        # with open(path_traj, 'wb') as file_traj:
        #     pkl.dump(self.traj, file_traj)

        self.cur_idx = 0
        self.step_counter = 0
        self.failure_counter = 0

        # self.traj = []
