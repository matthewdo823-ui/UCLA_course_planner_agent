"""Orchestrator entry point — the user-facing Agent instance.

This module creates the main Agent and includes the input_agent's protocol
so that incoming user ChatMessages are handled by the input agent's
conversational intake flow.  All other agents run as separate processes
(see run_all.py).
"""

from __future__ import annotations

from course_planner.agents.input_agent import agent, protocol  # noqa: F401

# The input_agent module already calls:
#   agent = Agent(name="input-agent", seed=..., port=8001, ...)
#   protocol = Protocol(spec=chat_protocol_spec)
#   agent.include(protocol, publish_manifest=True)
#
# So simply importing it is enough — `agent` is ready to run.

if __name__ == "__main__":
    agent.run()
