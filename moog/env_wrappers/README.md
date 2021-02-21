# Environment wrappers

The files in this directory contain wrappers to the [MOOG Environment
class](../environment.py). All wrappers (except the GymWrapper) preserve the API
of the Environment, and to this end they inherit from an [abstract base class
wrapper](./abstract_wrapper.py).

One might ask why not make the environment wrappers inherit from
../environment.Environment instead of wrapping and exposing all methods and
properties. The reason is because users may want to use multiple environment
wrappers in arbitrary combinations. If we had used inheritance we would not have
been able to accomodate these combinations of wrappers.

One might also ask why not use mixins. The reason is again to support using
multiple wrappers. Some of the wrappers override the same functions (e.g.
.__init__(), .step()), so disentangling them to operate as multiple mixins would
not be possible without sacrificing code cleanness/simplicity.

## Logger wrapper

We want to draw specific attention to the [logger wrapper](./logger.py) for
psychology and neurophysiology researchers. This can be used to record
environment state and subject actions.

However, the logger wrapper logs the full state and all sprite attributes, which
can yield large log files. Consequently, we recommend users to fork the logger
wrapper and modify it to only record the sprites and attributes that vary over
time in their task. For example, if in your task sprite colors never change,
then let the logger ignore the color channel attributes of sprites. Or if some
sprites (e.g. boundary walls) are unchanging and constant across episodes, let
the logger ignore those sprites.
