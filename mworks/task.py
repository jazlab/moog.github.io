"""Python task script for interfacing with MWorks.

The main class in this file is TaskManager, which is a task to be run by
main.mwel.
"""

################################################################################
####  Update sys.path
################################################################################

# MWorks clones python dependencies into some mworks-specific directory, so we
# need to manually set the sys.path here in order for imports to work. You will
# need to change the pwd and python site-packages here to point to the current
# directory and your computer's python site-packages (or that of a virtual
# environment).

_PWD = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/moog/mworks'
_PYTHON_SITE_PACKAGES = (
    '/Users/nicholaswatters/miniconda3/envs/mworks_moog/lib/python3.8/'
    'site-packages'
)

import sys

if '' not in sys.path:
    sys.path.insert(0, '')
if _PYTHON_SITE_PACKAGES not in sys.path:
    sys.path.append(_PYTHON_SITE_PACKAGES)
if _PWD not in sys.path:
    sys.path.append(_PWD)

################################################################################
####  Imports
################################################################################

from datetime import datetime
import importlib
import numpy as np
import os
import threading

from moog import action_spaces
from moog import environment
from moog import env_wrappers
from moog import observers


class TaskManager():
    """MOOG task manager.
    
    This loads a MOOG environment and handles the interface for MWorks. It
    should only be used by main.mwel.
    """

    def __init__(self, config_name='configs.pong', level=''):
        """Constructor.

        Args:
            config_name: String. Name of config file in configs/ to run.
            level: Argument for config_name.get_config().
        """
        self.lock = threading.Lock()

        config = importlib.import_module(config_name)
        # Force MWorks server to reload the config, so changes to the config
        # will be propagated to MWorks without restarting the server.
        importlib.reload(config)
        config = config.get_config(level)
        image_size = (getvar('image_pixel_width'), getvar('image_pixel_height'))
        renderer = observers.PILRenderer(
            image_size=image_size,
            anti_aliasing=1,
            color_to_rgb=config['observers']['image'].color_to_rgb,
        )
        config['observers'] = {'image': renderer}
        log_dir = os.path.join(
            _PWD, 'logs/' + datetime.now().strftime('%Y_%m_%d'),
            '/'.join(config_name.split('.')), 'level_' + str(level))
        self.env = env_wrappers.LoggingEnvironment(
            environment=environment.Environment(**config),
            log_dir=log_dir,
        )

        if isinstance(self.env.action_space, action_spaces.Joystick):
            self._using_joystick = True
        elif isinstance(self.env.action_space, action_spaces.Grid):
            self._using_joystick = False
        else:
            raise ValueError(
                'Unrecognized action space {}'.format(config['action_space']))
        
        self.complete = False

    def reset(self):
        """Reset environment.

        This should be called at the beginning of every trial.
        """
        self.env.reset()

        unregister_event_callbacks()
        self.events = {}
        
        if self._using_joystick:
            controller_varnames = ('x_force', 'y_force')
        else:
            controller_varnames = (
                'up_pressed', 'down_pressed', 'left_pressed', 'right_pressed')
        
        for varname in controller_varnames:
            self._register_event_callback(varname)

        self.complete = False

    def _register_event_callback(self, varname):
        self.events[varname] = []
        def cb(evt):
            with self.lock:
                self.events[varname].append(evt.data)
        register_event_callback(varname, cb)

    def _get_action_joystick(self):
        """Get joystick action."""
        if self.env.step_count == 0:
            # Don't move on the first step
            # We set x_force and y_force to zero because some joysticks
            # initially give a non-zero action, which persists unless we
            # explicitly terminate zero it out.
            setvar('x_force', 0.)
            setvar('y_force', 0.)
            return np.zeros(2)
        else:
            return np.array([getvar('x_force'), getvar('y_force')])
    
    def _get_action_grid(self):
        """Get grid action."""
        if self.env.step_count == 0:
            # Don't move on the first step
            # We set x_force and y_force to zero because some joysticks
            # initially give a non-zero action, which persists unless we
            # explicitly terminate zero it out.
            setvar('left_pressed', 0)
            setvar('right_pressed', 0)
            setvar('down_pressed', 0)
            setvar('up_pressed', 0)
            return 4
        else:
            keys_pressed = np.array([
                getvar('left_pressed'),
                getvar('right_pressed'),
                getvar('down_pressed'),
                getvar('up_pressed'),
            ])
            if sum(keys_pressed) > 1:
                keys_pressed[self._keys_pressed] = 0
            
            if sum(keys_pressed) > 1:
                random_ind = np.random.choice(np.argwhere(keys_pressed)[:, 0])
                keys_pressed = np.zeros(4, dtype=int)
                keys_pressed[random_ind] = 1
            
            self._keys_pressed = keys_pressed

            if sum(keys_pressed):
                key_ind = np.argwhere(keys_pressed)[0, 0]
            else:
                key_ind = 4
            
            return key_ind

    def step(self):
        """Step environment."""

        if self.complete:
            # Don't step if the task is already complete.  Returning None tells
            # MWorks that the image texture doesn't need to be updated.
            return

        if isinstance(self.env.action_space, action_spaces.Joystick):
            action = self._get_action_joystick()
        elif isinstance(self.env.action_space, action_spaces.Grid):
            action = self._get_action_grid()
        else:
            raise ValueError(
                'Unrecognized action space {}'.format(config['action_space']))

        timestep = self.env.step(action)
        reward = timestep.reward
        img = timestep.observation['image']

        if reward:
            setvar('reward_duration', reward * 1000)  # ms to us

        if timestep.last():
            setvar('end_trial', True)
            self.complete = True

        # MWorks' Python image stimulus requires a contiguous buffer, so we use
        # ascontiguousarray to provide one.
        to_return = np.ascontiguousarray(img)

        return to_return
