"""Environment wrapper class for running demo on multi-agent tasks.

The entry point is MultiAgentEnvironment, a class that behaves like a moog
environment but has agents living in it that take actions for all the agents
that the user is not controlling in the gui. So this effectively turns the
multi-agent environment into a single-agent demo-able environment.
"""

from moog import env_wrappers


class MultiAgentEnvironment(env_wrappers.AbstractEnvironmentWrapper):
    """Environment class for converting multi-agent into single-agent task."""
    
    def __init__(self, environment, agent_name, **other_agents):
        """Constructor.

        Args:
            environment: Instance of ../moog/environment.Environment. Should
                have a Composite action space.
            agent_name: Name of the agent being controlled by the player's
                joystick in the demo. Specifically, this is the key used for the
                joystick action value when passed into the Composite action
                space in the underlying environment.
            other_agents: Dictionary of agents. Keys must contain all the keys
                expected by the Composite action space except the agent_name
                key. Values are agent instances with a step(observation) method
                that returns an action.
        """
        super(MultiAgentEnvironment, self).__init__(environment)
        self._agent_name = agent_name
        self._other_agents = other_agents

    def step(self, action):
        """Step the environment with an action.
        
        This action should come from the demo. The actions for all other agents
        are filled in by stepping self._other_agents.
        """
        obs = self.observation()
        action_dict = {
            k: agent.step(obs) for k, agent in self._other_agents.items()
        }
        action_dict[self._agent_name] = action

        return self._environment.step(action_dict)
