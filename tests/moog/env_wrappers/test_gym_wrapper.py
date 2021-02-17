# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for moog/env_wrappers/gym_wrapper.py.

To run this test, navigate to this directory and run
```bash
$ pytest test_gym_wrapper.py --capture=tee-sys
```

Note: The --capture=tee-sys routes print statements to stdout, which is useful
for debugging.

Alternatively, to run this test and any others, navigate to any parent directory
and simply run
```bash
$ pytest --capture=tee-sys
```
This will run all test_* files in children directories.
"""

import sys
sys.path.insert(0, '...')  # Allow imports from moog codebase

import collections
from gym import spaces
import numpy as np

from moog import action_spaces
from moog import environment
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog.env_wrappers import gym_wrapper


class TestGymWrapper():

    def testJoystickActions(self):
        def _state_initializer():
            agent = sprite.Sprite(x=0.5, y=0.5, scale=0.1, c0=128)
            return collections.OrderedDict([('agent', [agent])])

        max_episode_length = 5
        task = tasks.Reset(
            lambda _: True, steps_after_condition=max_episode_length - 1)

        env = environment.Environment(
            state_initializer=_state_initializer,
            physics=physics_lib.Physics(),
            task=task,
            action_space=action_spaces.Joystick(),
            observers={'image': observers.PILRenderer(image_size=(64, 64))},
        )
        
        gym_env = gym_wrapper.GymWrapper(env)

        assert (
            gym_env.observation_space ==
            spaces.Dict({
                'image': spaces.Box(
                    -np.inf, np.inf, shape=(64, 64, 3), dtype=np.uint8)
            })
        )
        assert (
            gym_env.action_space ==
            spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        )

        for _ in range(3):
            gym_env.reset()
            for _ in range(max_episode_length - 1):
                action = gym_env.action_space.sample()
                obs, reward, done, _ = gym_env.step(action)
                assert (obs['image'].dtype == np.uint8)
                assert not done
                assert (reward == 0.)
            action = gym_env.action_space.sample()
            _, _, done, _ = gym_env.step(action)
            assert done
            _, _, done, _ = gym_env.step(action)
            assert not done

    def testGridActions(self):
        def _state_initializer():
            agent = sprite.Sprite(x=0.5, y=0.5, scale=0.1, c0=128)
            return collections.OrderedDict([('agent', [agent])])
        
        max_episode_length = 5
        task = tasks.Reset(
            lambda _: True, steps_after_condition=max_episode_length - 1)
        env = environment.Environment(
            state_initializer=_state_initializer,
            physics=physics_lib.Physics(),
            task=task,
            action_space=action_spaces.Grid(),
            observers={'image': observers.PILRenderer(image_size=(64, 64))})
        gym_env = gym_wrapper.GymWrapper(env)

        assert (
            gym_env.observation_space ==
            spaces.Dict({
                'image': spaces.Box(
                    -np.inf, np.inf, shape=(64, 64, 3), dtype=np.uint8)
            })
        )
        assert (gym_env.action_space == spaces.Discrete(5))

        for _ in range(3):
            gym_env.reset()
            for _ in range(max_episode_length - 1):
                action = gym_env.action_space.sample()
                obs, reward, done, _ = gym_env.step(action)
                assert (obs['image'].dtype == np.uint8)
                assert not done
                assert reward == 0.
            action = gym_env.action_space.sample()
            _, _, done, _ = gym_env.step(action)
            assert done
            _, _, done, _ = gym_env.step(action)
            assert not done
