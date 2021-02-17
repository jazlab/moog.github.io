"""Raw state observer."""

from . import abstract_observer


class RawState(abstract_observer.AbstractObserver):
  """Passes the raw state itself as observation.
  
  The raw state is an OrderedDict of iterables of sprites. Observing the raw
  state can be useful for example for hand-crafted agents with particular
  policies in multi-agent tasks.
  """

  def __init__(self):
    pass

  def __call__(self, state):
    """Passes the raw state straight through."""
    return state
  
  def observation_spec(self):
    raise NotImplementedError
