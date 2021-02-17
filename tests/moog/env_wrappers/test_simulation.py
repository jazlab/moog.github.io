"""Tests for moog/env_wrappers/simulation.py.

To run this test, navigate to this directory and run
```bash
$ pytest test_simulation.py --capture=tee-sys
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
import numpy as np

from moog import action_spaces
from moog import environment
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import sprite
from moog import tasks
from moog.env_wrappers import simulation


def get_env():
    """Get simple environment, wrapper in simulation wrapper."""
    def _state_initializer():
        agent = sprite.Sprite(x=0.5, y=0.5, scale=0.1, c0=128)
        target = sprite.Sprite(x=0.75, y=0.5, scale=0.1, c1=128)
        state = collections.OrderedDict(
            [('agent', [agent]), ('target', [target])])
        return state
    
    task = tasks.ContactReward(
        1., 'agent', 'target', reset_steps_after_contact=2)
    
    action_space = action_spaces.Grid(
        0.1, action_layers='agent', control_velocity=True)

    def _modify_meta_state(meta_state):
        meta_state['key'] = meta_state['key'] + 1
    update_meta_state = game_rules.ModifyMetaState(_modify_meta_state)

    env = environment.Environment(
        state_initializer=_state_initializer,
        physics=physics_lib.Physics(),
        task=task,
        action_space=action_space,
        observers={'image': observers.PILRenderer(image_size=(64, 64))},
        meta_state_initializer=lambda: {'key': 0},
        game_rules=(update_meta_state,),
    )
    sim_env = simulation.SimulationEnvironment(env)
    return sim_env


class TestSimulation():
    """Tests for simulation environment wrapper."""

    def testStep(self):
        """Test for normal stepping without simulation."""
        env = get_env()
        episode_actions = [1, 4, 3, 1, 2, 0]

        env.reset()
        for a in episode_actions[:-1]:
            timestep = env.step(a)
            assert not timestep.last()
        timestep = env.step(episode_actions[-1])
        assert timestep.last()

    def testSimStepSimPop(self):
        """Test for simulation stepping and popping."""
        env = get_env()
        sim_actions_init = [1, 4, 3]
        pop_inds_0 = [-1]
        sim_actions_reward_0 = [3, 1, 2, 0]
        pop_inds_1 = [-1, -2]
        sim_actions_reward_1 = [1, 2, 0]
        actions_reward = [1, 4, 3, 1, 2, 0]

        env.reset()
        
        # Simulate for a bit
        for a in sim_actions_init:
            timestep = env.sim_step(a)
            assert not timestep.last()
        
        # Pop out one step
        for i in pop_inds_0:
            env.sim_pop(i)
        
        # Simulate until reward
        for a in sim_actions_reward_0[:-1]:
            timestep = env.sim_step(a)
            assert not timestep.last()
        timestep = env.sim_step(sim_actions_reward_0[-1])
        assert timestep.last()

        # Pop a few steps
        for i in pop_inds_1:
            env.sim_pop(i)

        # Simulate until reward
        for a in sim_actions_reward_1[:-1]:
            timestep = env.sim_step(a)
            assert not timestep.last()
        timestep = env.sim_step(sim_actions_reward_1[-1])
        assert timestep.last()

        # Step until reward
        for a in actions_reward[:-1]:
            timestep = env.step(a)
            assert not timestep.last()
        timestep = env.step(actions_reward[-1])
        assert timestep.last()

        assert (env.meta_state['key'] == 7)
