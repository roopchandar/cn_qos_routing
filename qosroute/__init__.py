"""qosroute: DRL-based adaptive congestion-aware routing simulator.

Modules
-------
topology   : the router/link graph (NetworkX) and its physical properties
traffic    : flows, scenarios and traffic-load schedules
simulator  : discrete-time packet-level network simulator (1 tick = 1 ms)
metrics    : QoS metric definitions computed from a finished simulation
routing    : shortest-path, congestion-aware heuristic and DQN routers
agent      : DQN components (state encoder, Q-networks, replay buffer, agent)
"""

__version__ = "1.0.0"
