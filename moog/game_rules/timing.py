"""Rules that control the timing of when to apply other rules.

These are useful for when you want to delay a rule of apply a rule only in a
specific timeframe.

If you want your task to have trial structure, consider looking at the rules in
task_phases.py. Those are more amenable to creating tasks with phases, e.g.
fixation phase -> stimulus phase -> response phase -> reward phase.
"""

from . import abstract_rule
import numpy as np


class TimedRule(abstract_rule.AbstractRule):
    """Apply a set of rules only during a specified time interval.
    
    The time interval step_interval is given in terms of number of environment
    steps, i.e. number of times this rule is called. The upper bound (second
    element) of step_interval may be infinite. The step_interval may also be a
    callable returning random intervals.

    For example, if you want some rule my_rule to step only between 20 and 40
    environment steps (i.e. 20 to 40 timesteps after each reset), you can use
    TimedRule((20, 40), my_rule).
    """

    def __init__(self, step_interval, rules):
        """Constructor.

        Args:
            step_interval: Length-2 iterable [start_step_count, stop_step_count]
                or callable returning such. This range specifies the interval in
                which to step rules. The units of measurement are times this
                rule is called --- this usually corresponds to environment
                steps, but may be offset if this TimedRule itself is the
                argument of some other TimedRule.
            rules: Rule or iterable of rules to step during the step_interval.
        """
        if not callable(step_interval):
            self._step_interval = lambda: step_interval
        else:
            self._step_interval = step_interval
        
        if not isinstance(rules, (list, tuple)):
            rules = (rules,)
        self._rules = rules

    def reset(self, state, meta_state):
        self._steps_until_start, self._steps_until_stop = self._step_interval()
        for rule in self._rules:
            rule.reset(state, meta_state)

    def step(self, state, meta_state):
        """Apply rule to state."""
        if self._steps_until_start <= 0 and self._steps_until_stop > 0:
            for rule in self._rules:
                rule.step(state, meta_state)
        self._steps_until_start -= 1
        self._steps_until_stop -= 1


class DelayedRule(TimedRule):
    """TimedRule that starts at specified time and never stops."""

    def __init__(self, steps_until_start, rules, duration=np.inf):
        """Constructor.

        Args:
            steps_until_start: Int or callable returning Int. This is the time
                at which to start stepping rules.
            rules: Rule or iterable of rules.
            duration: Optional int or callable. Duration of the rule.
        """
        if not callable(steps_until_start):
            callable_steps_until_start = lambda: steps_until_start
        else:
            callable_steps_until_start = steps_until_start

        if not callable(duration):
            callable_duration = lambda: duration
        else:
            callable_duration = duration

        def _step_interval():
            start_time = callable_steps_until_start()
            return (start_time, start_time + callable_duration())
        
        super().__init__(_step_interval, rules)


class TemporaryRule(TimedRule):
    """TimedRule that starts immediately but stops after specified duration."""

    def __init__(self, steps_until_stop, rules):
        """Constructor.

        Args:
            steps_until_stop: Int or callable returning Int. This is the time at
                which to stop stepping rules.
            rules: Rule or iterable of rules.
        """
        if not callable(steps_until_stop):
            step_interval = lambda: (0, steps_until_stop)
        else:
            step_interval = lambda: (0, steps_until_stop())
        super().__init__(step_interval, rules)
