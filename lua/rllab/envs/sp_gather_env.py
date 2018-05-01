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

import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref

import numpy as np
import theano

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.envs.mujoco.gather.embedded_viewer import EmbeddedViewer
from rllab.envs.mujoco.mujoco_env import BIG
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjViewer, MjModel, mjcore, mjlib, \
    mjextra, glfw

from rllab.envs.mujoco.gather.gather_env import GatherViewer

APPLE = 0
BOMB = 1

MODEL_DIR = osp.abspath(osp.dirname(__file__))

class SPGatherEnv(Env, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    @autoargs.arg('n_apples', type=int,
                  help='Number of apples in each episode')
    @autoargs.arg('n_bombs', type=int,
                  help='Number of bombs in each episode')
    @autoargs.arg('activity_range', type=float,
                  help='The span for generating objects '
                       '(x, y in [-range, range])')
    @autoargs.arg('robot_object_spacing', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('catch_range', type=float,
                  help='Minimum distance range to catch an object')
    @autoargs.arg('n_bins', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('sensor_range', type=float,
                  help='Maximum sensor range (how far it can go)')
    @autoargs.arg('sensor_span', type=float,
                  help='Maximum sensor span (how wide it can span), in '
                       'radians')
    def __init__(
            self, opts,
            n_apples=8,
            n_bombs=8,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            *args, **kwargs
    ):
        self.opts = opts
        self.sp_state_coeffs = np.ones(13)
        if self.opts['sp_loc_only']:
            self.sp_state_coeffs[2:] = 0
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.objects = []
        super(SPGatherEnv, self).__init__(*args, **kwargs)
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        # pylint: disable=not-callable
        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        # pylint: enable=not-callable
        self.inner_env = inner_env
        Serializable.quick_init(self, locals())
        assert self.opts['naction_heads'] == self.inner_env.action_bounds[1].size + 1, \
            'number of action must be {}'.format(self.inner_env.action_bounds[1].size + 1)
        assert self.opts['rllab_cont_action'], 'set rllab_cont_action to true'
        assert self.inner_env.action_bounds[1][0] == self.opts['rllab_cont_limit'], \
            'set rllab_cont_limit to {}'.format(self.inner_env.action_bounds[1][0])
        self.total_step_count = 0
        self.total_step_count_test = 0

    def reset(self):
        # self-play stuff
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
        else:
            self.current_mind = 1
            self.stat['type'] = 'self_play'
            self.stat['self_play_count'] = 1

        # super(GatherMDP, self).reset()
        self.objects = []
        if self.test_mode:
            existing = set()
            while len(self.objects) < self.n_apples:
                x = np.random.randint(-self.activity_range / 2,
                                      self.activity_range / 2) * 2
                y = np.random.randint(-self.activity_range / 2,
                                      self.activity_range / 2) * 2
                # regenerate, since it is too close to the robot's initial position
                if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                    continue
                if (x, y) in existing:
                    continue
                typ = APPLE
                self.objects.append((x, y, typ))
                existing.add((x, y))
            while len(self.objects) < self.n_apples + self.n_bombs:
                x = np.random.randint(-self.activity_range / 2,
                                      self.activity_range / 2) * 2
                y = np.random.randint(-self.activity_range / 2,
                                      self.activity_range / 2) * 2
                # regenerate, since it is too close to the robot's initial position
                if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                    continue
                if (x, y) in existing:
                    continue
                typ = BOMB
                self.objects.append((x, y, typ))
                existing.add((x, y))

        self.inner_env.reset()
        self.target_obs = self.inner_env.get_current_obs()
        self.initial_pos = self.inner_env.get_current_obs().flat[:2].copy()
        self.init_full_state = self.inner_env._full_state.copy()
        return self.get_current_obs()

    def step(self, action_all):
        self.current_time += 1
        action = action_all[:2]
        _, _, done, info = self.inner_env.step(action)
        if done:
            assert False
            return Step(self.get_current_obs(), -10, done, **info)
        com = self.inner_env.get_body_com("torso")
        x, y = com[:2]
        reward = 0
        self.total_step_count += 1
        if self.test_mode:
            self.total_step_count_test += 1
            new_objs = []
            for obj in self.objects:
                ox, oy, typ = obj
                # object within zone!
                if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                    if typ == APPLE:
                        reward = reward + 1
                    else:
                        reward = reward - 1
                else:
                    new_objs.append(obj)
            self.objects = new_objs
            done = len(self.objects) == 0
            self.success = done
            if self.opts['sp_test_max_steps'] > 0:
                if self.current_time > self.opts['sp_test_max_steps']:
                    done = True
                    self.success = False
        else:
            reward = 0
            done = False
            self.success = False
            if self.current_mind == 1:
                if action_all[2] == 1:
                    self.switch_mind()
            else:
                target_state = np.multiply(self.sp_state_coeffs, self.target_obs)
                current_state = np.multiply(self.sp_state_coeffs, self.inner_env.get_current_obs())
                self.target_dist = np.linalg.norm(target_state - current_state)
                self.success = bool(self.target_dist < self.opts['sp_state_thres'])
                done = self.success
                if self.opts['sp_reward_bob_step']:
                    reward = -self.opts['sp_reward_coeff']

        return Step(self.get_current_obs(), reward, done, info=dict(current_mind = self.current_mind))

    def switch_mind(self):
        self.current_mind = 2
        self.switch_pos = self.inner_env.get_current_obs().flat[:2].copy()
        self.stat['switch_t'] = self.current_time
        self.stat['switch_dist'] = np.linalg.norm(self.switch_pos - self.initial_pos)
        self.stat['switch_count'] = 1
        if self.opts['sp_mode'] == 'repeat':
            self.target_obs = self.inner_env.get_current_obs()
            self.inner_env.reset(init_state=self.init_full_state)

    def get_readings(self):
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.inner_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins
        ori = self.get_ori()
        # print ori*180/math.pi
        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            #    ((angle + half_span) +
            #     ori) % (2 * math.pi) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.inner_env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()

        if self.test_mode:
            mode = 1
            target = np.zeros(13)
            time = 0
        else:
            mode = -1
            target = self.target_obs
            time = self.current_time/self.opts['max_steps']

        return np.concatenate([self_obs, apple_readings, bomb_readings, [mode, time], target])

    def reward_terminal(self):
        if not self.test_mode:
            return 0
        # return self.inner_env.get_body_com("torso")[0]
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
            if self.opts['sp_reward_bob_step']:
                return 0
            else:
                return self.opts['sp_reward_coeff'] * (self.stat['switch_t'] - self.current_time)

    def get_stat(self):
        if self.test_mode:
            self.stat['test_pos'] = self.inner_env.get_current_obs().flat[:2].copy()
            self.stat['test_steps'] = self.current_time
        else:
            self.stat['return_t'] = self.current_time - self.stat['switch_t']
            self.stat['switch_pos'] = self.switch_pos
            self.stat['final_pos'] = self.inner_env.get_current_obs().flat[:2].copy()
        self.stat['success'] = self.success
        return self.stat

    def get_viewer(self):
        if self.inner_env.viewer is None:
            self.inner_env.viewer = GatherViewer(self)
            self.inner_env.viewer.start()
            self.inner_env.viewer.set_model(self.inner_env.model)
        return self.inner_env.viewer

    @property
    @overrides
    def action_space(self):
        return self.inner_env.action_space

    @property
    def action_bounds(self):
        return self.inner_env.action_bounds

    @property
    def viewer(self):
        return self.inner_env.viewer

    @property
    @overrides
    def observation_space(self):
        dim = self.inner_env.observation_space.flat_dim
        newdim = dim + self.n_bins * 2
        ub = BIG * np.ones(newdim)
        return spaces.Box(ub * -1, ub)

    def action_from_key(self, key):
        return self.inner_env.action_from_key(key)

    def render(self):
        self.get_viewer()
        self.inner_env.render()

    def get_ori(self): # get orientation
        return self.inner_env.model.data.qpos[self.__class__.ORI_IND]
