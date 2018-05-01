# The MIT License (MIT)
# 
# Copyright (c) 2016 rllab contributors
# 
# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import os
import pygame
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

class SPMountainCarEnv(Box2DEnv, Serializable):

    @autoargs.inherit(Box2DEnv.__init__)
    @autoargs.arg("height_bonus_coeff", type=float,
                  help="Height bonus added to each step's reward")
    @autoargs.arg("goal_cart_pos", type=float,
                  help="Goal horizontal position")
    def __init__(self, opts,
                 height_bonus=1.,
                 goal_cart_pos=0.6,
                 *args, **kwargs):
        xmlpath = os.path.join(os.path.dirname(__file__), 'mountain_car.xml.mako')
        super(SPMountainCarEnv, self).__init__(
            xmlpath,
            *args, **kwargs
        )
        self.opts = opts
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        Serializable.quick_init(self, locals())
        self.total_step_count = 0
        self.total_step_count_test = 0

    @overrides
    def compute_reward(self, action):
        yield
        yield int(self.cart.position[0] >= self.goal_cart_pos) # like VIME paper

    @overrides
    def is_current_done(self):
        return self.cart.position[0] >= self.goal_cart_pos \
            or abs(self.cart.position[0]) >= self.max_cart_pos

    @overrides
    def get_current_obs(self):
        if self.test_mode:
            mode = 1
            target = np.zeros(2)
            time = 0
        else:
            mode = -1
            target = self.target_obs
            time = self.current_time/self.opts['max_steps']
        obs = super(SPMountainCarEnv, self).get_current_obs()
        return np.concatenate([obs, [mode, time], target])

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1],
            [1],
        ])
        low, high = bounds
        xvel = np.random.uniform(low, high)[0]
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])

        self.current_time = 0
        self.stat = dict()
        self.stat['switch_t'] = 0
        self.stat['switch_count'] = 0
        self.stat['switch_dist'] = 0
        self.switch_pos = np.zeros(2)
        self.test_mode = False
        if int(self.opts['nminds']) == 1:
            self.test_mode = True
        elif self.opts['sp_test_rate_bysteps']:
            if self.total_step_count_test < self.total_step_count * self.opts['sp_test_rate']:
                self.test_mode = True
        elif np.random.uniform() < self.opts['sp_test_rate']:
            self.test_mode = True

        if self.test_mode:
            self.current_mind = 2
            self.stat['type'] = 'test_task'
            self.stat['test_task_count'] = 1
            self.goal_position = self.goal_cart_pos
        else:
            self.current_mind = 1
            self.stat['type'] = 'self_play'
            self.stat['self_play_count'] = 1
            self.goal_position = 0
            self.initial_pos = self.cart.position[0]
            self.initial_vel = self.cart.linearVelocity[0]
            self.target_obs = super(SPMountainCarEnv, self).get_current_obs()

        return self.get_current_obs()

    @overrides
    def step(self, action_all):
        self.current_time += 1
        action = action_all[0]
        res = super(SPMountainCarEnv, self).step(action)
        obs = self.get_current_obs()

        done = False
        reward = 0
        self.total_step_count += 1
        if self.test_mode:
            self.total_step_count_test += 1
            done = res.done
            reward = res.reward
            self.success = bool(self.cart.position[0] >= self.goal_cart_pos)
        else:
            if res.done:
                done = res.done
                self.success = False
            else:
                if self.current_mind == 1 and action_all[1] == 1:
                    self.switch_mind()
                elif self.current_mind == 2:
                    self.success = bool(np.linalg.norm(self.target_obs - obs[:2]) < self.opts['sp_state_thres'])
                    done = self.success or abs(self.cart.position[0]) >= self.max_cart_pos

        return Step(observation=obs, reward=reward, done=done, info=dict(current_mind = self.current_mind))

    def switch_mind(self):
        self.current_mind = 2
        self.switch_pos = np.array([self.cart.position[0], self.cart.linearVelocity[0]])
        self.stat['switch_t'] = self.current_time
        self.stat['switch_dist'] = abs(self.cart.position[0] - self.initial_pos)
        self.stat['switch_count'] = 1
        self.goal_position = self.cart.position[0]
        self.target_obs = super(SPMountainCarEnv, self).get_current_obs()
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        self.cart.linearVelocity = (self.initial_vel, self.cart.linearVelocity[1])

    def reward_terminal(self):
        return 0

    def reward_terminal_mind(self, mind):
        if self.test_mode:
            return 0

        if mind == 1:
            if self.current_mind == 2:
                return self.opts['sp_reward_coeff'] * max(0, (self.current_time - 2 * self.stat['switch_t']))
            else:
                return 0
        else:
            return self.opts['sp_reward_coeff'] * (self.stat['switch_t'] - self.current_time)

    def get_stat(self):
        if self.test_mode:
            self.stat['test_pos'] = np.array([self.cart.position[0], self.cart.linearVelocity[0]])
            self.stat['test_steps'] = self.current_time
        else:
            self.stat['return_t'] = self.current_time - self.stat['switch_t']
            self.stat['switch_pos'] = self.switch_pos
            self.stat['final_pos'] = np.array([self.cart.position[0], self.cart.linearVelocity[0]])
        self.stat['success'] = self.success
        return self.stat

    @overrides
    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-1])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+1])
        else:
            return np.asarray([0])
