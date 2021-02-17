"""Multi-agent cleanup task config with hand-crafted agents.

This imports the cleanup task config from moog/configs/examples_multi_agent, but
adds hand-crafted agents to it for the demo to run alongside the user.
"""

from moog_demos.example_configs import cleanup
from multi_agent_example.handcrafted_agents import cleanup as cleanup_agents


def get_config(level):
    """Get config.
    
    The config is the cleanup config from moog/configs/examples_multi_agent with
    additional 'agent_name' and 'agents' fields that are needed for the demo to
    recognise and be able to run the multi-agent task. See
    ../../moog_demos/run_demo.py for how the demo does this.
    """

    config = cleanup.get_config(level)
    config['agent_name'] = 'agent_0'
    config['agents'] = {
        'agent_1': cleanup_agents.SelfishAgent(name='agent_1'),
        'agent_2': cleanup_agents.FickleAgent(
            cleanup_agents.SelfishAgent(name='agent_2'),
            cleanup_agents.SelflessAgent(name='agent_2'),
            steps_per_agent=100,
        ),
    }

    return config
