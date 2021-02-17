"""Gif writer to record a video while playing the demo.

Note: If the enter key prints `^M` instead of entering the input, run the
following command:
$ stty sane
"""

import imageio
import logging
import numpy as np
import os
import sys


class GifWriter(object):
    """GifWriter class.
    
    Usage:
        my_gif_writer = GifWriter('path/to/a/file.gif')
        for image in my_video:
            my_gif_writer.add(image)
        my_gif_writer.close()
    """

    def __init__(self, gif_file, fps=5):
        """Constructor.

        Args:
            gif_file: String. Full path to gif filename. Should end in '.gif'.
            fps: Int. Frames per second for the gif.
        """

        # If the gif directory does not exist, ask the user if they want to
        # create it.
        gif_file = os.path.expanduser(gif_file)
        gif_dir = os.path.dirname(gif_file)
        if not os.path.exists(gif_dir):
            print('Directory {} does not exist'.format(gif_dir))
            should_create = input(
                'Would you like to create that directory?  (y/n)')
            if should_create == 'y':
                print('Creating directory {}'.format(gif_dir))
                os.makedirs(gif_dir)
            else:
                print('exiting')
                sys.exit()

        # If the gif directory already exists, ask the user if they want to
        # override it.
        if os.path.isfile(gif_file):
            print('File {} to write gif to already exists.'.format(gif_file))
            should_override = input(
                'Would you like to override the file there?  (y/n)')
            if should_override == 'y':
                print('Removing {}'.format(gif_file))
                os.remove(gif_file)
            else:
                print('exiting')
                sys.exit()
                
        self._gif_file = gif_file
        self._images = []
        self._fps = fps

    def add(self, image):
        self._images.append(image)

    def close(self):
        """Write the gif."""
        print('Writing gif with {} images to file {}'.format(
            len(self._images), self._gif_file))
        imageio.mimsave(self._gif_file, self._images, fps=self._fps)
