# Intrinsic Motivation via Self-play
This is the code we used for our paper [Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](https://arxiv.org/abs/1703.05407). Some parts of the code are adopted from other projects:

* The code in `games/` is adapted from [MazeBase](https://github.com/facebook/MazeBase).
* Some files in `rllab/` are adapted from [RLLab](https://github.com/rll/rllab).

## Setup
The code is written in **Lua**, so you would need that. Also, install [Torch7](https://github.com/torch/distro) and [Visdom](https://github.com/facebookresearch/visdom) (only needed for plotting) packages. Since the code is multi-threaded, a machine with many cores is recommended. Don't forget to set `OMP_NUM_THREADS=1`. If you have less than 20 cores, set `--nworker` to smaller value, but increase `--batch_size` at the same time to ensure the total batch size (=`nworker`x`batch_size`) is unchanged. 

For running RLLab experiments, you need to install it from [here](https://rllab.readthedocs.io/en/latest/). Also, you would need a [Mujoco](http://www.mujoco.org/) license for running SwimmerGather experiments.

You would need [TorchCraft](https://github.com/TorchCraft/TorchCraft) to run StarCraft experiments. In particular, we used this [branch](https://github.com/TorchCraft/TorchCraft/tree/eb7453db4ae1e43403b54aaca69af6f139b7ca0a). Also you need to configure it to use `starcraft/economy.scm` map.


## Mazebase experiment

For a fast test run:
```bash
th -i main.lua --games_config_path games/config/sp_lightkey_small.lua \
   --visibility 5 --max_steps 40 \
   --nminds 2 --mind_reward_separate \
   --encoder_lut --encoder_lut_size 30 --mind_target \
   --sp_mode reverse --entropy_reg 0.003 --sp_test_rate 0.2 \
   --show --plot
```
Should learn in 100 epochs.

## RLLab experiments
To run a MountainCar experiment:
```bash
th -i main.lua --nactions 5:2 --rllab \
   --rllab_env SPMountainCar --rllab_in_dim 6 \
   --max_steps 500  \
   --nworker 8 \
   --epochs 50 --rllab_cont_action \
   --entropy_reg 0.003 --nminds 2 --mind_reward_separate \
   --sp_test_rate 0.1 \
   --sp_state_thres 0.2 \
   --show --plot
```

To run a SwimmerGather experiment:
```bash
th -i main.lua --max_steps 200 \
   --nminds 2 --mind_reward_separate \
   --sp_mode reverse --entropy_reg 0.003 --sp_test_entr_zero \
   --sp_test_max_steps 166 --sp_loc_only --sp_state_thres 0.3 \
   --sp_test_rate 0.1 --show --plot --epochs 390 --rllab \
   --rllab_env SPSwimmerGather --rllab_in_dim 48 --rllab_cont_action \
   --nactions 9:9:2 --rllab_cont_limit 50 \
   --show --plot
```

## StarCraft experiments
First, launch 32 StarCraft instances with consecutive ports starting from 11111. Then run 
```bash
th -i main.lua --nworker 16 --batch_size 2 --sc \
    --max_steps 200 --max_info 12 --coop \
    --nactions 7 --nagents 10 --nbatches 1 --visibility 8 \
    --nminds 2 --mind_reward_separate --sp_mode repeat \
    --sp_test_rate 0.1 --entropy_reg 0.003 --sp_test_entr_zero \
    --encoder_lut --encoder_lut_size 100 --epochs 500 \
    --show --plot
```
