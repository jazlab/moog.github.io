"""Restore and plot logged data.

This file can be used to visualize data logged by
..moog.env_wrappers.logger.LoggingEnvironment.

To run, first play .run_demo.py with flag --log_data=True. Then find the log
directory that was written to, which will be printed at the beginning of playing
the demo and should be of the form 'logs/config_name/$integer/$date_time'. Then
using that directory, run
$ python3 restore_logged_data.py --config='path.to.your.config' \
  --log_directory=your_log_directory

You may need to also set the --level flag.
"""
import sys
sys.path.insert(0, '..')

from absl import app
from absl import flags

import collections
import importlib
import json
import logging
import numpy as np
import os
from moog import observers
from moog import sprite as sprite_lib

from matplotlib import path as mpl_path
from matplotlib import pyplot as plt
from matplotlib import transforms as mpl_transforms

FLAGS = flags.FLAGS
flags.DEFINE_string('config',
                    'example_configs.pong',
                    'Filename of task config to use.')
flags.DEFINE_integer('level', 0, 'Level of task config to run.')
flags.DEFINE_string('log_directory',
                    'logs/pong/0/2021_02_06_16_18_58',
                    'Directory of logs to restore.')
flags.DEFINE_integer('num_episodes', 3, 'Number of episodes to restore.')


def _create_new_sprite(sprite_kwargs, vertices=None):
    """Create new sprite from factors.

    Args:
        sprite_kwargs: Dict. Keyword arguments for sprite_lib.Sprite.__init__().
            All of the strings in sprite_lib.Sprite.FACTOR_NAMES must be keys of
            sprite_kwargs.
        vertices: Optional numpy array of vertices. If provided, are used to
            define the shape of the sprite. Otherwise, sprite_kwargs['shape'] is
            used.

    Returns:
        Instance of sprite_lib.Sprite.
    """
    if vertices is not None:
        # Have vertices, so must invert the translation, rotation, and
        # scaling transformations to get the original sprite shape.
        center_translate = mpl_transforms.Affine2D().translate(
            -sprite_kwargs['x'], -sprite_kwargs['y'])
        x_y_scale = 1. / np.array([
            sprite_kwargs['scale'],
            sprite_kwargs['scale'] * sprite_kwargs['aspect_ratio']
        ])
        transform = (
            center_translate +
            mpl_transforms.Affine2D().rotate(-sprite_kwargs['angle']) +
            mpl_transforms.Affine2D().scale(*x_y_scale)
        )
        vertices = mpl_path.Path(vertices)
        vertices = transform.transform_path(vertices).vertices

        sprite_kwargs['shape'] = vertices

    return sprite_lib.Sprite(**sprite_kwargs)


def _state_str_to_image(state_str, renderer, attributes, stored_sprites):
    """Convert state string to image.

    Args:
        state_str: String. This should be a state string from an episode log.
            It should contain a list of layers, each of which is of the form
            [layer_name, sprites_attributes] where sprites_attributes is a list
            of attributes of all the sprites in the layer.
        renderer: Instance of observers.PILRenderer. Used to renderer the state.
        attributes: List of strings. Each must be either an element of
            sprite_lib.Sprite.FACTOR_NAMES of 'id'.
        stored_sprites: Dict. Keys are unique id values of sprites, and values
            are sprite instances. This dictionary is dynamically updated to keep
            track of sprites so they don't need to be re-instantianted every
            step.

    Returns:
        Image of rendered state.
    """
    state = collections.OrderedDict()

    # Will keep track of which sprite ids are still in use
    active_sprite_ids = []

    for layer_name, layer_str in state_str:
        layer_sprites = []

        for sprite_str in layer_str:
            create_new_sprite = False
            vertices = None

            if len(sprite_str) == len(attributes) + 1:
                # Vertices are the last element of sprite_str
                vertices = np.array(sprite_str.pop(-1))
                create_new_sprite = True
            elif len(sprite_str) != len(attributes):
                raise ValueError(
                    'len(sprite_str) = {} must be equal to or one greater than '
                    'len(attributes) = {}'.format(
                        len(sprite_str), len(attributes)))

            # Kwargs for the constructor of sprite_lib.Sprite()
            sprite_kwargs = {
                k: v for k, v in zip(attributes, sprite_str)
            }

            # All attributes other than 'id' should be in
            # sprite_lib.Sprite.FACTOR_NAMES
            sprite_id = sprite_kwargs.pop('id')
            active_sprite_ids.append(sprite_id)

            if sprite_id not in stored_sprites:
                create_new_sprite = True

            # Create new sprite if necessary, else update old sprite
            if create_new_sprite:
                sprite = _create_new_sprite(sprite_kwargs, vertices=vertices)
                stored_sprites[sprite_id] = sprite
            else:
                sprite = stored_sprites[sprite_id]
                sprite_lib.update_sprite(sprite, **sprite_kwargs)

            layer_sprites.append(sprite)
        state[layer_name] = layer_sprites

    # Purge stored_sprites, removing all inactive sprites
    inactive_ids = [k for k in stored_sprites if k not in active_sprite_ids]
    [stored_sprites.pop(k) for k in inactive_ids]

    return renderer(state)


def main(_):
    """Restore and render logged data."""
    log_dir = FLAGS.log_directory

    ############################################################################
    #### Create renderer
    ############################################################################

    # Use the PILRenderer in config['observers'] if it exists
    config = importlib.import_module(FLAGS.config)
    config = config.get_config(FLAGS.level)
    renderer_observer = False
    for renderer in config['observers'].values():
        if isinstance(renderer, observers.PILRenderer):
            renderer_observer = True
            break
    if not renderer_observer:
        # config['observers'] has no PILRenderer, so use this one and hope for
        # the best
        renderer = observers.PILRenderer(
            image_size=(256, 256),
            anti_aliasing=1,
            color_to_rgb='hsv_to_rgb',  # Depends on the config
        )

    logging.info('Renderer instantiated.')

    ############################################################################
    #### Load logged episodes
    ############################################################################

    log_filenames = sorted(
        filter(lambda s: s.isnumeric(), os.listdir(log_dir)))
    log_filenames = log_filenames[:min(FLAGS.num_episodes, len(log_filenames))]
    episode_strings = [
        json.load(open(os.path.join(log_dir, x))) for x in log_filenames]
    attributes = json.load(open(os.path.join(log_dir, 'attributes.txt')))

    logging.info('Episode strings read.')

    ############################################################################
    #### Render logged episodes
    ############################################################################

    episode_images = []
    stored_sprites = {}
    for i, episode_str in enumerate(episode_strings):
        logging.info('Rendering episode {}'.format(i))
        for timestep in episode_str:
            state_str = timestep[-1]
            image = _state_str_to_image(
                state_str, renderer, attributes, stored_sprites)
            episode_images.append(image)

    ############################################################################
    #### Display video of logged episodes
    ############################################################################

    logging.info('Displaying rendered episodes in infinite loop.')
    ax = plt.subplots()[1]
    imshow = ax.imshow(episode_images[0])

    index = 0
    while True:
        index = (index + 1) % len(episode_images)
        imshow.set_data(episode_images[index])
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    app.run(main)
