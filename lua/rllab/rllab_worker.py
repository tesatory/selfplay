import sys
import numpy as np
from rllab.envs.normalized_env import NormalizedEnv

sys.argv = []
sys.argv.append("RLLab")

def init(env_name, size, opts):
    global envs
    envs = []
    for i in range(int(size)):
        if env_name == 'SPSwimmer':
            from envs.sp_swimmer_env import SPSwimmerEnv
            envs.append(SPSwimmerEnv())
        elif env_name == 'SPSwimmerGather':
            from envs.sp_swimmer_gather_env import SPSwimmerGatherEnv
            envs.append(SPSwimmerGatherEnv(opts))
        elif env_name == 'SPMountainCar':
            from envs.sp_mountain_car import SPMountainCarEnv
            envs.append(SPMountainCarEnv(opts))
        elif env_name == 'Swimmer':
            from rllab.envs.mujoco.swimmer_env import SwimmerEnv
            envs.append(SwimmerEnv())
        else:
            raise RuntimeError("wrong env name")
        if opts['rllab_normalize_rllab']:
            envs[-1] = NormalizedEnv(env=envs[-1], normalize_obs=True)

def reset():
    obs = []
    for i in range(len(envs)):
        obs.append(envs[i].reset())
    return obs

def step(action, active, steps):
    state, reward, done, info = [], [], [], []
    for i in range(len(envs)):
        if int(active[i]) == 0:
            state.append(0)
            reward.append(0)
            done.append(True)
            info.append(dict())
        else:
            s, r, d, f = None, None, None, None
            for s in range(int(steps)):
                s, r, d, f = envs[i].step(action[i])
                if d:
                    break
            state.append(s)
            reward.append(r)
            done.append(d)
            if 'info' in f:
                info.append(f['info'])
            else:
                info.append(f)
    return (state, reward, done, info)

def reward_terminal():
    reward = []
    for i in range(len(envs)):
        if hasattr(envs[i], 'reward_terminal'):
            reward.append(envs[i].reward_terminal())
        else:
            reward.append(0)
    return reward

def render(get_image = False):
    envs[0].render()
    if get_image:
        data, w, h = envs[0].get_viewer().get_image()
        return np.fromstring(data, dtype='uint8').reshape(h, w, 3)[::-1, :, :]

def obs_shape():
    return envs[0].observation_space.shape

def num_actions():
    return envs[0].action_space.n

def get_stat():
    stat = []
    for i in range(len(envs)):
        if hasattr(envs[i], 'get_stat'):
            stat.append(envs[i].get_stat())
        else:
            stat.append(dict())
    return stat

# below for self-play training only

def reward_terminal_mind(mind):
    reward = []
    for i in range(len(envs)):
        if hasattr(envs[i], 'reward_terminal_mind'):
            reward.append(envs[i].reward_terminal_mind(mind[i]))
        else:
            reward.append(0)
    return reward

def current_mind():
    mind = []
    for i in range(len(envs)):
        mind.append(envs[i].current_mind)
    return mind
