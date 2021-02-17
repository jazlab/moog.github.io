"""Run all example configs, using random actions.

To run this test, navigate to this directory and run
```bash
$ pytest test_examples.py --capture=tee-sys
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
sys.path.insert(0, '../../..')  # Allow imports from moog codebase

import importlib
import logging
from moog import environment
from moog import env_wrappers
import pytest

_NUM_EPISODES = 2  # Number of episodes per config to run
_NUM_STEPS = 200  # Number of steps per episode to run

# Dictionary from config name to iterable of all levels for that config
_CONFIG_LEVELS = {
    'bounce_box_contact_prediction': [True, False],
    'chase_avoid_torus': [0, 1],
    'colliding_predators': [None],
    'falling_balls': [None],
    'first_person_predators_prey': [None],
    'functional_maze': [None],
    'match_to_sample': [2, 3, 4],
    'multi_tracking_with_feature': [2, 3, 4],
    'pacman': [0, 1],
    'parallelogram_catch': [0, 1, 2],
    'predators_arena': [1, 2, 3],
    'red_green': [0, 1, 2, 3],
    'pong': [None],
}


class TestExampleConfigs():
    """Run all the example configs and levels in _CONFIG_LEVELS."""

    @pytest.mark.parametrize(
        'config_name',
        list(_CONFIG_LEVELS.keys()),
    )
    def testExampleConfigs(self, config_name):
        config_module = importlib.import_module(
            'moog_demos.example_configs.' + config_name)
        for level in _CONFIG_LEVELS[config_name]:
            config = config_module.get_config(level)
            env = environment.Environment(**config)
            for _ in range(_NUM_EPISODES):
                env.reset()
                for _ in range(_NUM_STEPS):
                    env.step(action=env.action_space.random_action())

    def testMultiAgentExample(self):
        config_name = 'multi_agent_example.configs.cleanup'
        config_module = importlib.import_module(config_name)
        config = config_module.get_config(None)
        agents = config.pop('agents')
        agent_name = config.pop('agent_name')
        multi_env = environment.Environment(**config)
        env = env_wrappers.MultiAgentEnvironment(
            environment=multi_env, agent_name=agent_name, **agents)

        for _ in range(_NUM_EPISODES):
            env.reset()
            for _ in range(_NUM_STEPS):
                action_space = env.action_space.action_spaces[agent_name]
                env.step(action=action_space.random_action())

