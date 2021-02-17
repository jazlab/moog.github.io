"""Rule that is applied only when a condition is satisfied.

The main class here is ConditionalRule, which applies a rule or rules X number
of times, where X is a bool or an integer returned by a function applied to the
state.
"""

from . import abstract_rule
import inspect


class ConditionalRule(abstract_rule.AbstractRule):
    """Apply a rule X number of times, where X is a function of the state.

    X may be a boolean, indicating whether to apply the rule once or not.

    For example, here is code for a rule that creates a new prey every time a
    prey is contacted by an agent:
    ```python
        my_rule = ConditionalRule(
            condition=get_contact_counter('prey', 'agent'),
            rules=CreateSprites(
                layer='prey',
                generator=prey_generator,
                without_overlapping=['walls', 'agent']),
        )
    ```
    """

    def __init__(self, condition, rules):
        """Constructor.

        Args:
            condition: Function with one of the following signatures:
                    * state --> bool or int
                    * state, meta_state --> bool or int
                If output is bool, indicates whether to apply rule. If output is
                int, indicates how many times to apply rule.
            rules: Instance of abstract_rule.AbstractRule or iterable of such.
        """
        if len(inspect.signature(condition).parameters.values()) == 1:
            self._condition = lambda state, meta_state: condition(state)
        else:
            self._condition = condition
        
        if not isinstance(rules, (list, tuple)):
            self._rules = [rules]
        else:
            self._rules = rules

    def reset(self, state, meta_state):
        for rule in self._rules:
            rule.reset(state, meta_state)

    def step(self, state, meta_state):
        for _ in range(self._condition(state, meta_state)):
            for rule in self._rules:
                rule.step(state, meta_state)
