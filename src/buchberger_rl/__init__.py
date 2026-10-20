"""Learning S-pair selection strategies for Buchberger's algorithm.

Subpackages
-----------
algebra     exact polynomial arithmetic over GF(p) and a steppable Buchberger algorithm
strategies  classical and look-ahead pair-selection heuristics
ideals      random distributions of binomial ideals
env         the Markov decision process and pair featurisation
agent       pair-scoring policy network and PPO (NumPy, explicit gradients)
evaluation  paired, statistically tested comparisons on held-out ideals
"""

__version__ = "1.0.0"
