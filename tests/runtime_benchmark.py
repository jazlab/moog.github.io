"""Run benchmarks and print benchmark report.

This file times various aspects of the environment, such as the physics engine
and the renderer, given a task config. It is useful to benchmark new task
configs.

Note: To run this file, you must install the tqdm package.
"""
import sys
sys.path.insert(0, '..')  # Allow imports from parent directory

from absl import flags
from absl import app
import importlib
import numpy as np
import time
from tqdm import tqdm

from moog import environment
from moog.observers import pil_renderer
from moog.observers import color_maps

FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    'moog.configs.examples.pacman',
                    'Filename of task config to use.')
flags.DEFINE_integer('level', 0, 'Level of task config to run.')
flags.DEFINE_string('color_map', 'hsv_to_rgb',
                    'Color map in observers/color_maps.py to use.')

_IMAGE_SIZE_ANTI_ALIASING = (
    (64, 1),
    (128, 1),
    (256, 1),
    (512, 1),
    (512, 2),
    (1024, 1),
)

_NUM_RESETS = 20
_TRIALS_PER_RESET = 20


def _time_env_function(env, env_function):
    times_list = []
    for reset_count in tqdm(range(_NUM_RESETS)):
        env.reset()
        t_start = time.time()
        for _ in range(_TRIALS_PER_RESET):
            abort = env_function()
            if abort:
                break
        if abort:
            continue
        t_end = time.time()
        times_list.append(t_end - t_start)
    ms_per_step = 1e3 * np.array(times_list) / float(_NUM_RESETS)
    print('  ms/step:  {}'.format(np.mean(ms_per_step)))
    print('  stddev ms/step:  {}'.format(np.std(ms_per_step)))
    print('  min ms/step:  {}'.format(np.min(ms_per_step)))
    print('  max ms/step:  {}'.format(np.max(ms_per_step)))


def main(_):
    """Run benchmarking script."""
    
    config = importlib.import_module(FLAGS.config)
    print('Benchmarking config:  {}'.format(FLAGS.config))
    config = config.get_config(FLAGS.level)

    ############################################################################
    # Benchmark without rendering, using random actions
    ############################################################################

    config['observers'] = {'image': lambda _: None}
    env = environment.Environment(**config)
    print('Environment steps without rendering:')
    def _step_env_function():
        obs = env.step(action=env.action_space.random_action())
        if obs.last():
            return True
        else:
            return False
    _time_env_function(env, _step_env_function)

    ############################################################################
    # Benchmark only resets, without rendering, using random actions
    ############################################################################

    config['observers'] = {'image': lambda _: None}
    env = environment.Environment(**config)
    print('Environment resets, without rendering:')
    def _step_env_function():
        obs = env.reset()
    _time_env_function(env, _step_env_function)

    ############################################################################
    # Benchmark physics only
    ############################################################################

    config['observers'] = {'image': lambda _: None}
    env = environment.Environment(**config)
    print('Physics steps only:')
    def _physics_env_function():
        env.physics.step(env.state)
        return False
    _time_env_function(env, _physics_env_function)

    ############################################################################
    # Benchmark renderer only
    ############################################################################

    def _get_render_env_function(env):
        def _render_env_function():
            env.observation()
            return False
        return _render_env_function

    for image_size, anti_aliasing in _IMAGE_SIZE_ANTI_ALIASING:
        renderer = pil_renderer.PILRenderer(
            image_size=(image_size, image_size),
            anti_aliasing=anti_aliasing,
            color_to_rgb=getattr(color_maps, FLAGS.color_map)
        )
        config['observers'] = {'image': renderer}
        env = environment.Environment(**config)

        print('Renderer steps only, image_size {}, anti_aliasing {}:'.format(
            image_size, anti_aliasing))
        _time_env_function(env, _get_render_env_function(env))

    ############################################################################
    # Benchmark full steps with rendering
    ############################################################################

    def _get_env_function(env):
        def _env_function():
            obs = env.step(action=env.action_space.random_action())
            if obs.last():
                return True
            else:
                return False
        return _env_function

    for image_size, anti_aliasing in _IMAGE_SIZE_ANTI_ALIASING:
        renderer = pil_renderer.PILRenderer(
            image_size=(image_size, image_size),
            anti_aliasing=anti_aliasing,
            color_to_rgb=getattr(color_maps, FLAGS.color_map)
        )
        config['observers'] = {'image': renderer}
        env = environment.Environment(**config)

        print(
            'Full steps with rendering, image_size {}, anti_aliasing '
            '{}:'.format(image_size, anti_aliasing))
        _time_env_function(env, _get_env_function(env))


if __name__ == "__main__":
    app.run(main)
