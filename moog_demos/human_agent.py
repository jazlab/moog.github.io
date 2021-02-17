"""Human gui to demo tasks.

This contains tools for an interactive gui controlled by a human. The main class
is HumanAgent, which takes and environment and runs it in an iteractive display
that shows the rendered environement, plot of recent rewards, and a
human-playable action space (e.g. cartoon joystick, arrow keys, etc.).

This is useful for testing task prototypes.

Note: When recording gifs, be sure to pressing the 'esc' key when you want to
stop the demo and write the gif.

Note: If the reward plot does not appear in your figure window, that is probably
because your monitor screen is not tall enough to fit it, given the rendering
size you chose. Consider using a smaller render_size in the renderer (eg. 256).
"""

import logging
import math
import numpy as np
import sys
import time
import tkinter as tk

import mss
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageTk
from matplotlib.backends import backend_tkagg

from moog_demos import gui_frames
from moog import action_spaces

_WINDOW_ASPECT_RATIO = 2.7  # height/width for the gui window


class HumanAgent():
    """Human-playable agent.

    This provides a gui for human interaction with an environment. The gui
    displays the rendered environment, a plot of rewards over recent history,
    and a frame which may contain a joystick, depending on the action space.

    Note: This agent does not abide by the standard RL agent api in that it does
    not have a .step() method taking an observation and returning an action.
    This is because due to Tkinter's interactive .mainloop() function, the
    entire environment interaction must be implemented as a callback in the gui.
    """

    def __init__(self, env, render_size, fps=10, reward_history=20,
                 observation_image_key='image', gif_writer=None,
                 reward_border_width=10):
        """Constructor.

        Args:
            env: instance of moog.environment.Environment. The environment
                observations are assumed to have a key 'image' whose value is an
                image with height and width render_size.
            render_size: Int. Width and height of observation_image_key value of
                the environment observation image.
            fps: Int. Frames per second to run, if possible. Note: The tkinter
                gui interface and matplotlib rendering is slow, often taking
                about 40 milliseconds for 256x256 rendering. Furthermore,
                depending on the environment and physics, stepping the
                environent can take some time (usually no more than 10
                milliseconds). Therefore, the fastest fps this gui can run is
                20fps for 256x256 rendering, and if the given fps is higher than
                this it will be capped. This slowness is mostly due to
                matplotlib and tkinter, not the environment or rendering itself,
                and does not occur when training agents or using mworks for
                psychophysics.
            reward_history: Int. Number of history timesteps to plot the reward
                for.
            observation_image_key: String. Key of the observation that is the
                image.
            gif_writer: Optional instance of a gif writer. This writer should
                have a .close() method and an .add(frame) method.
            reward_border_width: Int. Width of the red/green border to render
                when a positive/negative reward is given.
        """
        self._env = env
        self._ms_per_step = 1000. / fps
        self._reward_history = reward_history
        self._observation_image_key = observation_image_key
        self._gif_writer = gif_writer
        self._reward_border_width = reward_border_width

        # This will be used later to capture a section of the screen
        if self._gif_writer:
            self._screen_capture = mss.mss()

        # Create root Tk window and fix its size
        self.root = tk.Tk()
        frame_width = str(render_size)
        frame_height = str(int(_WINDOW_ASPECT_RATIO * render_size))
        self.root.geometry(frame_width + 'x' + frame_height)

        # Bind escape key to terminate gif_writer and exit
        def _escape(event):
            if self._gif_writer:
                self._gif_writer.close()
            sys.exit()

        self.root.bind('<Escape>', _escape)

        ########################################################################
        # Create the environment display and pack it into the top of the window.
        ########################################################################

        image = env.reset().observation[self._observation_image_key]
        self._env_canvas = tk.Canvas(
            self.root, width=render_size, height=render_size)
        self._env_canvas.pack(side=tk.TOP)
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self._env_canvas.img = img
        self.image_on_canvas = self._env_canvas.create_image(
            0, 0, anchor="nw", image=self._env_canvas.img)

        ########################################################################
        # Create the gui frame and pack it into the bottom of the window.
        ########################################################################

        canvas_half_width = render_size / 2
        action_space = env.action_space
        if isinstance(action_space, action_spaces.Composite):
            num_agents = len(action_space.action_spaces)
            a_spaces = action_space.action_spaces.values()
            if (num_agents == 2 and all(
                    isinstance(a, action_spaces.Grid) for a in a_spaces)):
                logging.info(
                    '2-player Grid action space. One player uses arrow keys '
                    'and the other uses [a, s, d, w].')
                # Two-player Grid action spaces
                self.gui_frame = gui_frames.TwoPlayerGridActions(
                    self.root,
                    canvas_half_width=canvas_half_width,
                    player_0=action_space.action_keys[0],
                    player_1=action_space.action_keys[1],
                )
            elif (num_agents == 2 and all(
                    isinstance(a, action_spaces.Joystick) for a in a_spaces)):
                logging.info(
                    '2-player Joystick action space. One player uses arrow keys '
                    'and the other uses [a, s, d, w].')
                # Two-player Joystick action spaces
                self.gui_frame = gui_frames.TwoPlayerJoystick(
                    self.root,
                    canvas_half_width=canvas_half_width,
                    player_0=action_space.action_keys[0],
                    player_1=action_space.action_keys[1],
                )
            else:
                logging.info(
                    'Composite action space provided, human controls only the '
                    'first agent.')
                action_space = list(action_space.action_spaces.values())[0]

        if isinstance(action_space, action_spaces.Joystick):
            logging.info(
                'Joystick action space, drag and move the joystick at the '
                'bottom of the window.')
            self.gui_frame = gui_frames.JoystickFrame(
                self.root,
                canvas_half_width=canvas_half_width,
                motion_zone_radius=canvas_half_width - 5,
            )
        elif isinstance(action_space, action_spaces.Grid):
            logging.info('Grid action space, use arrow keys.')
            self.gui_frame = gui_frames.GridActions(
                self.root,
                canvas_half_width=canvas_half_width,
            )
        elif isinstance(action_space, action_spaces.SetPosition):
            logging.info('SetPosition action space, click on the frame to act.')
            self.gui_frame = gui_frames.SetPositionFrame(
                self._env_canvas,
                canvas_half_width=canvas_half_width,
            )
        elif not isinstance(action_space, action_spaces.Composite):
            raise ValueError(
                'Cannot demo action space {}'.format(env.action_space))

        if not isinstance(action_space, action_spaces.SetPosition):
            self.gui_frame.canvas.pack(side=tk.BOTTOM)

        ########################################################################
        # Create the reward plot and pack it into the middle of the window.
        ########################################################################

        # This figuresize and the subplot adjustment is hand-crafted to make it
        # look okay for _WINDOW_ASPECT_RATIO 2.7. Ideally, we would infer the
        # space available for this reward plot and size the figure accordingly,
        # but I could not easily figure out how to do that --- it seems like
        # matplotlib doesn't count axis ticks and labels as part of the figure
        # size so those are cut off without handcrafting some subplot
        # adjustment.
        fig = plt.Figure(figsize=(2, 2), dpi=100)
        fig.subplots_adjust(bottom=0.5, left=0.25, right=0.95)

        self._ax_reward = fig.add_subplot(111)
        self._ax_reward.set_ylabel('Reward')
        self._ax_reward.set_xlabel('Time')
        self._ax_reward.axhline(y=0.0, color='lightgrey')

        # Plot rewards as a bar plot
        self._reset_rewards()
        reward_ticks = np.arange(-1 * self._reward_history + 1, 1)
        self.rewards_plot = self._ax_reward.bar(reward_ticks, self._rewards)

        # Create canvas in which to draw the reward plot and pack it
        # A tk.DrawingArea.
        self.rewards_canvas = backend_tkagg.FigureCanvasTkAgg(
            fig, master=self.root)
        self.rewards_canvas.draw()
        self.rewards_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)

        ########################################################################
        # Start run loop, automatically running the environment.
        ########################################################################

        self.root.after(math.floor(self._ms_per_step), self.step)
        self.root.mainloop()

    def _reset_rewards(self):
        self._rewards = np.zeros(self._reward_history)
        self._reward_range = [-1e-3, 1e-3]

    def render(self, observation):
        """Render the environment display and reward plot."""
        # Put green border if positive reward, red border if negative reward
        observation_image = observation.observation[self._observation_image_key]

        if self._reward_border_width:
            # Add a red/green border to the image for positive/negative reward.
            if sum(self._rewards[-4:]) > 0:
                show_border = True
                border_color = np.array([0, 255, 0], dtype=np.uint8)
            elif sum(self._rewards[-4:]) < 0:
                show_border = True
                border_color = np.array([255, 0, 0], dtype=np.uint8)
            else:
                show_border = False

            if show_border:
                observation_image[:self._reward_border_width] = border_color
                observation_image[-self._reward_border_width:] = border_color
                observation_image[:, :self._reward_border_width] = border_color
                observation_image[:, -self._reward_border_width:] = border_color

        # Set the image in the environment display to the new observation
        self._env_canvas.img = ImageTk.PhotoImage(Image.fromarray(
            observation_image))
        self._env_canvas.itemconfig(
            self.image_on_canvas, image=self._env_canvas.img)

        # Set the reward plot data to the current self._rewards
        for rect, h in zip(self.rewards_plot, self._rewards):
            rect.set_height(h)
            rect.set_facecolor('g' if h > 0 else 'r')
        self.rewards_canvas.draw()
        self._ax_reward.set_ylim(*self._reward_range)

    def step(self):
        """Take an action in the environment and render."""
        step_start_time = time.time()
        action = self.gui_frame.action  # action from the gui
        observation = self._env.step(action)

        # Update rewards and reward_range
        self._rewards[:-1] = self._rewards[1:]
        self._rewards[-1] = observation.reward
        if observation.reward:
            self._reward_range[0] = min(
                self._reward_range[0], observation.reward)
            self._reward_range[1] = max(
                self._reward_range[1], observation.reward)

        # display new observation
        self.render(observation)
        if observation.last():
            self._reset_rewards()

        # Screengrab the window and update the gif_writer
        if self._gif_writer:
            window = {
                'top': self.root.winfo_rooty(),
                'left': self.root.winfo_rootx(),
                'width': int(self.root.winfo_width()),
                'height': int(self.root.winfo_height()),
            }
            img = np.asarray(self._screen_capture.grab(window))
            # For some reason screen capture switches red and blue color
            # channels
            img = img[:, :, [2, 1, 0, 3]]
            self._gif_writer.add(img)

        # Recurse to step again after self._ms_per_step milliseconds
        step_end_time = time.time()
        delay = (step_end_time - step_start_time) * 1000  # convert to ms
        self.root.after(math.floor(self._ms_per_step - delay), self.step)
