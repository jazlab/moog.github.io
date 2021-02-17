"""Environment wrapper class for logging episodes.

This can be used to record data from a subject playing the task. See
../../moog_demos/restore_logged_data.py for an example of how to read log files.

Note: This logger records everything about the environment, which can be a lot
of data (depending on the task). If you plan to use this at scale for recording
subjects' or agents' behavior, we recommend forking this and modifying it to
only log the data that you need to do analyses for your specific task. For
example you may not want to log the positions/velocities of static sprites
(e.g. walls), or may not want to log all the attributes of sprites every
timestep (e.g. if you know that the colors of the sprites don't change in your
task).
"""

import copy
from datetime import datetime
import json
import logging
import numpy as np
import os
import time

from moog import env_wrappers
from moog import sprite

# This is the number of numerals in filenames. Since there is one file per
# episode, you should pick _FILENAME_ZFILL large enough that the number of
# episodes in your dataset is less than 10^_FILENAME_ZFILL.
_FILENAME_ZFILL = 5


class VertexLogging():
    NEVER = 'NEVER'
    ALWAYS = 'ALWAYS'
    WHEN_NECESSARY = 'WHEN_NECESSARY'


def _serialize(x):
    """Serialize a value x.

    This is used to serialize sprite attributes, actions, and meta_state so that
    they are json-writable.

    Specifically, numpy arrays are not JSON serializable, so we must convert
    numpy arrays to lists. This function is recursive to handle nestings inside
    of lists/tuples/dictionaries.

    Args:
        x: Value to serialize.

    Returns:
        Serialized value that can be JSON dumped.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (np.int32, np.int64)):
        return int(x)
    elif isinstance(x, list):
        return [_serialize(a) for a in x]
    elif isinstance(x, tuple):
        return (_serialize(a) for a in x)
    elif isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    else:
        return x


class LoggingEnvironment(env_wrappers.AbstractEnvironmentWrapper):
    """Environment class for logging timesteps.

    This logger produces a description of the log in 'description.txt' of
    log_dir, so please refer to that for a detailed account of the structure of
    the logs.
    """

    def __init__(self, environment, log_dir='logs',
                 log_vertices='WHEN_NECESSARY'):
        """Constructor.

        Args:
            environment: Instance of ../moog/environment.Environment.
            log_dir: String. Log directory relative to working directory.
            log_vertices: String. Of the following options:
                * 'NEVER'. In this case, never log sprite vertices.
                * 'WHEN_NECESSARY'. In this case, log sprite vertices when a
                    sprite has either just appeared or just changed shape. In
                    this way, the vertices of a sprite can always be inferred
                    from the current position/angle/aspect_ratio and the
                    vertices that were logged for that sprite (identifiable by
                    its id) the last time its vertices were logged.
                * 'ALWAYS'. Log vertices for all sprites every timestep.
        """
        super(LoggingEnvironment, self).__init__(environment)

        # Make sure log_vertices is a valid value
        if not hasattr(VertexLogging, log_vertices):
            raise ValueError('log_vertices is {} but must be in VertexLogging '
                             'values'.format(log_vertices))
        self._log_vertices = log_vertices

        # Set the logging directory
        now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if log_dir[0] == '/':
            log_dir = os.path.join(log_dir, now_str)
        else:
            log_dir = os.path.join(os.getcwd(), log_dir, now_str)
        os.makedirs(log_dir)
        self._log_dir = log_dir

        # These are the attributes that we'll log
        self._attributes = list(sprite.Sprite.FACTOR_NAMES) + ['id']

        # Log attribute list
        attributes_filename = os.path.join(self._log_dir, 'attributes.txt')
        logging.info('Logging attribute list {} to {}.'.format(
            self._attributes, attributes_filename))
        with open(attributes_filename, 'w') as f:
            json.dump(self._attributes, f)

        # Log description
        self._log_description()

        # Initialize self._episode_log
        self._episode_count = 0
        self._episode_log = []

    def _log_description(self):
        """Log a description of the data to a description.txt file."""
        description_filename = os.path.join(self._log_dir, 'description.txt')
        logging.info('Logging description to {}.'.format(description_filename))
        description = (
            'Each numerical file in this directory is an episode of the task. '
            'Each such file contains a json-serialized list, each element of '
            'which represents an environment step in the episode. Each step is '
            'a list of four elements, [[`time`, time], [`reward`, reward], '
            '[`step_type`, step_type], [`action`, action], [`meta_state`, '
            'meta_state`], state].'
            '\n\n'
            '\n\n'
            'time is a timestamp of the timestep.'
            '\n\n'
            '\n\n'
            'reward contains the value of the reward at that step.'
            '\n\n'
            '\n\n'
            'step_type indicates the dm_env.StepType of that step, i.e. '
            'whether it was first, mid, or last.'
            '\n\n'
            '\n\n'
            'action contains the agent action for the step.'
            '\n\n'
            '\n\n'
            'meta_state is the serialized meta_state of the environment.'
            '\n\n'
            '\n\n'
            'state is a list, each element of which represents a layer in the '
            'environment state. The layer is represented as a list [k, [], [], '
            '[], ...], where k is the layer name and the subsequent elements '
            'are serialized sprites. Each serialized sprite is a list of '
            'attributes. See attributes.txt for the attributes contained.'
        )
        if self._log_vertices == VertexLogging.ALWAYS:
            description += (
                ' Furthermore, a list of vertices is appended to the attribute '
                'list for each serialized sprite.'
            )
        elif self._log_vertices == VertexLogging.WHEN_NECESSARY:
            description += (
                '\n\n'
                '\n\n'
                'Furthermore, a list of vertices is appended to the attribute '
                'list for a serialized for the first timestep in which that '
                'serialized sprite appears, or when the sprite has changed '
                'shape.'
            )
        with open(description_filename, 'w') as f:
            f.write(description)

    def _serialize_sprite(self, s):
        """Serialize a sprite as a list of attributes."""
        attributes = [_serialize(getattr(s, x)) for x in self._attributes]
        if (self._log_vertices == VertexLogging.ALWAYS or
            (self._log_vertices == VertexLogging.WHEN_NECESSARY and
                s.just_set_shape)):
            attributes.append(s.vertices.tolist())
        s.just_set_shape = False
        return attributes

    def _serialized_state(self):
        """Serialized a state."""
        serialized_state = [
            [k, [self._serialize_sprite(s) for s in self.state[k]]]
            for k in self.state
        ]
        return serialized_state

    def step(self, action):
        """Step the environment with an action, logging timesteps."""
        timestep = self._environment.step(action)
        str_timestep = (
            [['time', time.time()],
             ['reward', timestep.reward],
             ['step_type', timestep.step_type.value],
             ['action', _serialize(action)],
             ['meta_state', _serialize(self._environment.meta_state)],
             self._serialized_state()]
        )
        self._episode_log.append(str_timestep)
        
        if timestep.last():
            # Write the episode to a log file
            episode_count_str = str(self._episode_count).zfill(_FILENAME_ZFILL)
            filename = os.path.join(self._log_dir, episode_count_str)
            logging.info('Logging episode {} to {}.'.format(
                self._episode_count, filename))
            with open(filename, 'w') as f:
                json.dump(self._episode_log, f)
            self._episode_count += 1
            self._episode_log = []

        return timestep
