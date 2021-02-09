# Flowcomm
Code for "Learning Correlated Communication Topology in Multi-Agent Reinforcement Learning" experiment in multiagent particle world for the amended cooperative navigation task "simple_spread_local" and "simple_spread_hetero".

We use MAAC (Iqbal, et al) as our base algorithm and integrate our graph modele to the original algorithm.

To run the code, first install the MPE in the particle_env folder, type shell command "pip install -e .".

Then, in the base directory, type "python main.py simple_spread_local maac", the default number of agents is 4. For the hetero task, the default number of agents is 8.
