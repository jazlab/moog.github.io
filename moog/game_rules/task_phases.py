"""Rules to facilitate tasks with phases.

This rules are useful if you want trials to have phasic structure, e.g.
fixation phase -> stimulus phase -> response phase -> reward phase,
with each transition between phases controlled by either some condition or a
temporal duration.
"""

from . import abstract_rule
import inspect
import numpy as np


class Phase():
    """Phase rule.
    
    This rule applies one-time-rules the first step, then continual rules for
    all subsequent steps until either an end condition is met or a timeout
    duration is reached.

    This is usually used as part of PhaseSequence() (see below).
    """

    def __init__(self,
                 one_time_rules=(),
                 continual_rules=(),
                 end_condition=None,
                 duration=np.inf,
                 name=''):
        """Constructor.

        Args:
            one_time_rules: Rule or iterable of rules, to be applied once the
                first time this phase is stepped.
            continual_rules: Rule or iterable of rules, to be applied each step
                during this phase.
            end_condition: Function with one of the following signatures:
                    * state --> bool
                    * state, meta_state --> bool
                Output bool is whether phase should end. Default is always
                False.
            duration: Int. maximum duration of phase.
            name: String. Optional name for the phase. This is sometimes used by
                PhaseSequence().
        """
        if not isinstance(one_time_rules, (list, tuple)):
            self._one_time_rules = (one_time_rules,)
        else:
            self._one_time_rules = one_time_rules

        if not isinstance(continual_rules, (list, tuple)):
            self._continual_rules = (continual_rules,)
        else:
            self._continual_rules = continual_rules
        
        if end_condition is None:
            self._end_condition = lambda state, meta_state: False
        elif len(inspect.signature(end_condition).parameters.values()) == 1:
            self._end_condition = lambda state, meta_state: end_condition(state)
        else:
            self._end_condition =  end_condition

        if not callable(duration):
            self._duration =  lambda: duration
        else:
            self._duration = duration

        self._name = name

    def reset(self, state, meta_state):
        """Reset at beginning of episode."""
        for rule in list(self._one_time_rules) + list(self._continual_rules):
            rule.reset(state=state, meta_state=meta_state)
        self._should_end = False
        self._step_count = 0
        self._current_duration = self._duration()

    def step(self, state, meta_state):
        """Step rule on environment state and meta_state."""
        if self.should_end:
            return
        
        if self._step_count == 0:
            for rule in self._one_time_rules:
                rule.step(state=state, meta_state=meta_state)
        
        for rule in self._continual_rules:
            rule.step(state=state, meta_state=meta_state)
        
        self._step_count += 1

        if (self._step_count >= self._current_duration or
                self._end_condition(state, meta_state)):
            self._should_end = True

    @property
    def should_end(self):
        return self._should_end
    
    @property
    def name(self):
        return self._name


class PhaseSequence():
    """PhaseSequence rule.
    
    This rule applies multiple Phase rules in sequence, applying each after the
    previous one ends.
    """

    def __init__(self, *single_phases, meta_state_phase_name_key=None):
        """Constructor.

        Args:
            *single_phases: Instances of Phase() (see above).
            meta_state_phase_name_key: Optional string. If given, the
                environment meta_state will contain the name of the current
                phase in this key, assuming it is a dictionary.
        """
        self._phases = single_phases
        self._meta_state_key = meta_state_phase_name_key

    def reset(self, state, meta_state):
        for phase in self._phases:
            phase.reset(state=state, meta_state=meta_state)
        self._current_phase_ind = 0
        self._current_phase = self._phases[0]
        if self._meta_state_key is not None:
            meta_state[self._meta_state_key] = self._current_phase.name

    def step(self, state, meta_state):
        if self._current_phase_ind >= len(self._phases):
            pass
        
        self._current_phase.step(state=state, meta_state=meta_state)
        if self._current_phase.should_end:
            self._current_phase_ind += 1
            self._current_phase = self._phases[self._current_phase_ind]
            if self._meta_state_key is not None:
                meta_state[self._meta_state_key] = self._current_phase.name
