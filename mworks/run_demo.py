"""Run demo.

This script can be used to play with and test prototype tasks with a
keyboard/mouse interface. It is nearly the same as moog_demos/run_demo.py.

Run with the following:
$ python3 run_demo.py --config=path.to.your.config

See also the flags at the top of this file. There are options for rendering
settings, recording to gif, and logging the behavior.

To exit the demo on a mac, press the 'esc' key. To customize key bindings, see
the key bindings in human_agent.py. If you are playing a task with an action
space that is not supported by the interfaces in gui_frames.py, please add a
custom gui interface.
"""

import sys
sys.path.append('..')

from absl import app
from absl import flags
import importlib
import os

from moog_demos import human_agent
from moog import observers
from moog import environment

FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    'configs.pong',
                    'Filename of task config to use.')
flags.DEFINE_integer('level', 0, 'Level of task config to run.')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 1, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('fps', 500, 'Frames per second.')
flags.DEFINE_integer('reward_history', 30,
                     'Number of historical reward timesteps to plot.')
flags.DEFINE_boolean('fixation_phase', False,
                     'Whether to include fixation phase.')


def main(_):
    """Run interactive task demo."""
    config = importlib.import_module(FLAGS.config)
    config = config.get_config(FLAGS.level)

    config['observers']['image'] = observers.PILRenderer(
        image_size=(FLAGS.render_size, FLAGS.render_size),
        anti_aliasing=FLAGS.anti_aliasing,
        color_to_rgb=config['observers']['image'].color_to_rgb,
        polygon_modifier=config['observers']['image'].polygon_modifier,
    )
    env = environment.Environment(**config)
    
    # Constructing the agent automatically starts the environment
    human_agent.HumanAgent(
        env,
        render_size=FLAGS.render_size,
        fps=FLAGS.fps,
        reward_history=FLAGS.reward_history,
        gif_writer=None,
    )


if __name__ == "__main__":
    app.run(main)
