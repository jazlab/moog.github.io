# MWorks with Modular Object-Oriented Games

## Summary

This directory contains an example of how to run MOOG with an
[Mworks](https://mworks.github.io/) interface. MWorks is a platform for
psychology and neurophysiology tasks.

While MWorks lacks flexibility for interactive physics-based tasks, it provides
precise stimulus timing control and interfaces with eye-trackers,
electrophysiology software, and a variety of controllers. Running MOOG
under-the-hood from MWorks as the example in this directory does combines the
best of both worlds. Furthermore, by implementing your task in MOOG you can run
python-based RL agents on the same task as your experimental subjects without
having to mainting two implementations.

## Getting Started

This directory only serves as an example of how to run MOOG from MWorks.

To run this example, follow these steps:

1. Install MWorks. Currently (as of 02/07/2021), you need the "bleeding edge"
   nightly build of MWorks, which you can install by downloading the "Nighly
   Build" on the [MWorks downloads page](https://mworks.github.io/downloads/). 
2. Create a virtual environment with python version 3.8. If you are using conda,
   this can be done with `conda create -n your_env_name python=3.8`
3. Activate your newly created virtual environment and install MOOG. See
   "Getting Started" in the [main README](../README.md).
4. Copy this directory onto your computer. You can use `git clone` and remove
   the other directories or copy the files in this directory manually (there are
   only a few files).
5. To check that the python task runs, run `$ python3 run_demo.py` and make sure
   that the pong task begins playing.
6. In [`task.py`](task.py), edit `PWD` to reflect the current working directory
   and edit `_PYTHON_SITE_PACKAGES` to reflect the path to your new virtual
   environment's python site-packages.
7. Launch the MWorks Server application (MWServer) and the MWorks Client
   application (MWClient). In the client, connect to the server by clicking the
   red cross and pressing "connect". This should result in a green checkmark ---
   if it doesn't, check that the listening port and listening address in the
   MWorks server match the URL and port in the client connection.
8. Load the [main.mwel](main.mwel) experiment by clicking on the experiment
   folder in the MWorks client and choosing that file for a new experiment.
9. In the MWorks client, select the "pong" experiment and press the green play
   button. You should then be able to play the pong task with the arrow keys on
   your keyboard.
10. You can also select the "pacman" task instead of pong to play that one.
    However, to play pacman you must first include `'interfaces/keyboard_grid'`
    in [main.mwel](main.mwel) instead of `'interfaces/keyboard_joystick'` (see
    the commented lines near the top of [main.mwel](main.mwel)).

If you have a gamepad (e.g. a PS3 controller), you can also plug that into your
computer's USB port and use it's rocker. Just be sure to includer the
appropriate controller interface at the top of [main.mwel](main.mwel), namely
`'interfaces/gamepad_joystick'` for pong or `'interfaces/gamepad_grid'` for
pacman.

## Running Your Tasks

To run your own tasks, begin by implementing your own task configs in MOOG. We
recommend also testing your task configs in python (by running
[`run_demo.py`](run_demo.py) on them) before trying them in MWorks, because
MWorks does not support the [python
debugger](https://docs.python.org/3/library/pdb.html) and does not display print
statements from python.

Once your task config is implemented, add a protocol for it at the bottom of
[main.mwel](main.mwel). Then you should be able to run it from MWorks.

If your task uses an action space other than `moog.action_spaces.Joystick` or
`moog.action_spaces.Grid`, then you will have to modify [`task.py`](task.py) to
handle the new kind of action space.

## Tips, Tricks, and Troubleshooting

MWorks interfaces:

* If MWorks does not recognize your keyboard, you might have to change the
  `preferred_location_id` in the `keyboard_device` in
  [`keyboard_joystick.mwel`](interfaces/keyboard_joystick.mwel). On a mac you
  can look up your keyboard's location ID in "About This Mac" --> "System
  Report" --> "USB". If you are using a bluetooth keyboard, this might not work
  and you should switch to a USB keyboard or a gamepad.
* If MWorks does not recognize your gamepad, you can similarly look up the
  gamepad's location ID and use that in
  [`gamepad_joystick.mwel`](interfaces/gamepad_joystick.mwel) or
  [`gamepad_grid.mwel`](interfaces/gamepad_grid.mwel).
* If you'd like to use different keys on your keyboard or gamepad, you can find
  their usage numbers by setting `log_all_input_values = True` in the device
  interface script, playing the game, pressing the keys you'd like to use, and
  looking at the outputs in the MWorks console.
* If you'd like to use other types of interfaces (e.g. joystick, eye tracker,
  etc.), please refer to the MWorks documentation and write your own interface
  files.

Debugging:

* MWorks does not pipe print statements in python to its console and does not
  support the python debugger. This can make debugging your python code from
  MWorks difficult. For this reason, we recommend testing your task configs with
  [`run_demo.py`](run_demo.py) as much as possible.
* If you are encountering a python bug when running MWorks (e.g. a bug coming
  from [`task.py`](task.py)), we recommend creating a local dump file in the
  python code and writing to that as a surrogate for print statements.
* Also note that MWorks has `report` statements that can be useful for debugging
  in MWorks.

## Recording Environment and Behavior Data

MWorks records event logs. These will contain events for all variable changes in
the MWorks script. See the [MWorks data analysis
documentation](https://mworks.github.io/documentation/latest/guide/data_analysis)
for more details.

However, the MWorks logs will not record the display images themselves nor the
state of MOOG. To record the state of MOOG, the MOOG environment in [`task.py`]
uses a [`LoggerEnvironment` wrapper](../moog/env_wrappers/logger.py), which logs
the state of the MOOG environment at every timestep. See
[`restore_logged_data.py`](../moog_demos/restore_logged_data.py) for and example
of how to read those logs. However, the `LoggerEnvironment` wrapper logs all
possible data from the MOOG environment, which can make the log files rather
large, so we recommend you write a custom logger wrapper that only logs the
state data you need for your task.
