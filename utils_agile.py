import numpy as np
import torch
import os
import pickle
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, data, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.data = data
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        states = self.data[0][episode_id]
        actions = self.data[1][episode_id]
        original_action_shape = actions.shape
        episode_len = states.shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        bpos = states[start_ts, :3]  # TODO: need to change these for pingpong
        bvel = states[start_ts, 3:6]
        qpos = states[start_ts, 6:9]
        qvel = states[start_ts, 9:]

        # get all actions after and including start_ts
        action = actions[start_ts:]
        action_len = episode_len - start_ts

        # Below is the original codes hack for real ALOHA
        # action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
        # action_len = episode_len - max(0, start_ts - 1)  # hack, t

        self.is_sim = True # Kin Man : hardcode
        padded_action = torch.zeros(original_action_shape, dtype=torch.float32)
        padded_action[:action_len] = action
        is_pad = torch.zeros(episode_len)
        is_pad[action_len:] = 1

        # construct observations
        bpos_data = bpos.float()
        bvel_data = bvel.float()
        qpos_data = qpos.float()
        qvel_data = qvel.float()
        action_data = padded_action.float()
        is_pad = is_pad.bool()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        bpos_data = (bpos_data - self.norm_stats["bpos_mean"]) / self.norm_stats["bpos_std"]
        bvel_data = (bvel_data - self.norm_stats["bvel_mean"]) / self.norm_stats["bvel_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]

        return bpos_data, bvel_data, qpos_data, qvel_data, action_data, is_pad


def get_norm_stats(data, dataset_name, num_episodes):
    all_qpos_data = data[0][..., 6:9]
    all_qvel_data = data[0][..., 9:]
    all_bpos_data = data[0][..., :3]
    all_bvel_data = data[0][..., 3:6]
    all_action_data = data[1]
    example_qpos = data[0][0,0,6:9]

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # normalize qvel data
    qvel_mean = all_qvel_data.mean(dim=[0, 1], keepdim=True)
    qvel_std = all_qvel_data.std(dim=[0, 1], keepdim=True)
    qvel_std = torch.clip(qvel_std, 1e-2, np.inf)  # clipping

    # normalize bpos data
    bpos_mean = all_bpos_data.mean(dim=[0, 1], keepdim=True)
    bpos_std = all_bpos_data.std(dim=[0, 1], keepdim=True)
    bpos_std = torch.clip(bpos_std, 1e-2, np.inf)  # clipping

    # normalize bvel data
    bvel_mean = all_bvel_data.mean(dim=[0, 1], keepdim=True)
    bvel_std = all_bvel_data.std(dim=[0, 1], keepdim=True)
    bvel_std = torch.clip(bvel_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "qvel_mean": qvel_mean.numpy().squeeze(), "qvel_std": qvel_std.numpy().squeeze(),
             "bpos_mean": bpos_mean.numpy().squeeze(), "bpos_std": bpos_std.numpy().squeeze(),
             "bvel_mean": bvel_mean.numpy().squeeze(), "bvel_std": bvel_std.numpy().squeeze(),
             "example_qpos": example_qpos}

    return stats


def load_data(dataset_dir, dataset_name, num_episodes, horizon, max_path_length, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    dataset_path = os.path.join(dataset_dir, dataset_name)
    print("Loading pickle file: ", dataset_path)
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

        states_all = []
        actions_all = []
        for traj in data:
            states = np.array([t[0] for t in traj], dtype=np.float32)
            actions = np.array([t[1] for t in traj], dtype=np.float32)
            actions = actions.reshape(-1, 6)
            preferences = traj[0][6]  # all preferences are the same per trajectory (unused here)

            path_length = len(states)
            if path_length > max_path_length:
                raise ValueError(
                    f"Path length: {path_length} is greater than max trajectory length: {max_path_length}")

            if horizon > path_length:
                # pad all the trajectories up to the horizon length at least
                states = np.pad(states, ((0, horizon - path_length), (0, 0)), 'edge')
                actions = np.pad(actions, ((0, horizon - path_length), (0, 0)), 'edge')
            elif horizon < path_length:
                # truncate the data outside of horizon
                states = states[:horizon]
                actions = actions[:horizon]

            states_all.append(states)
            actions_all.append(actions)
        states_all = torch.from_numpy(np.array(states_all))
        actions_all = torch.from_numpy(np.array(actions_all))
        processed_data = [states_all, actions_all]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(processed_data, dataset_name, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, processed_data, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, processed_data, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
