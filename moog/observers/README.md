# Observers

The [MOOG Environment class](../environment.py) takes an `observer` keyword
argument that is a dictionary of observer instances. These observers are called
at every step to create an observation for the timestep (i.e. an observation of
the environment to pass to the subject).

Observers must satisfy the API of [`AbstractObserver`](./abstract_observer.py).
This directory currently only contains two observers, a [PIL-base
renderer](./pil_renderer.py) and an [observer that returns the raw environment
state](./raw_state.py). All other files are supporting file for those.

However, as with all other components in MOOG, you are welcome to implement your
own custom observers in your configs (or in separate files that your configs
import), just be sure to inherit from `AbstractObserver`.
