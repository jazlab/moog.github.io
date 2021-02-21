# Tasks

The [MOOG Environment class](../environment.py) takes a `task` keyword argument
that defines a reward structure and trial reset condition. This directory
contains some task classes that can be used.

Tasks must satisfy the API of [`AbstractTask`](./abstract_task.py).

## Composite Task

Of particular note is the [`CompositeTask`](./composite_task.py), which allows a
user to compose tasks, summing their rewards. For example, if you want a task
that gives a reward when the agent contacts a target and a reward at regular
intervals and resets upon some other condition, you can compose a
[`ContactReward`](./contact_reward.py) task, a [`StayAlive`](./stay_alive.py)
task, and a [`Reset`](./reset.py) task in a `CompositeTask`.
