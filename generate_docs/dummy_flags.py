"""Dummy flags.

This script is used only to make the doc auto-generation script work. Since
multiple files in the MOOG library import flags and some of them use the same
flag (e.g. `--config` is used multiple times), an ImportError is raised when
pdoc imports the files, which causes pdoc to fail to generate documentation for
some files. To work around this, in generate_docs.sh we temporarily replace
imports of absl.flags by imports of this dummy flag file, which mimics the API
but does not raise import errors upon duplicate flags.
"""

FLAGS = None
DEFINE_string = lambda x, y, z: None
DEFINE_integer = lambda x, y, z: None
DEFINE_boolean = lambda x, y, z: None
