
import pickle
import numpy as np
import os

reward_path = r"C:\Users\felix\TmrlData\reward\reward.pkl"

if not os.path.exists(reward_path):
    print(f"Error: {reward_path} not found.")
else:
    with open(reward_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded reward.pkl. Type: {type(data)}")
    if isinstance(data, list):
        data = np.array(data)
    
    print(f"Shape: {data.shape}")
    if len(data) > 0:
        print(f"First point: {data[0]}")
        print(f"Last point: {data[-1]}")
        
        # Calculate bounding box
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        print(f"Bounding Box:\n Min: {min_vals}\n Max: {max_vals}")
        
        # Calculate average distance between points
        diffs = np.linalg.norm(data[1:] - data[:-1], axis=1)
        print(f"Avg distance between points: {np.mean(diffs):.4f}")
        print(f"Max distance between points: {np.max(diffs):.4f}")
        print(f"Total length: {np.sum(diffs):.4f}")
