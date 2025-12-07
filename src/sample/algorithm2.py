"""
Algorithm 2: Gibbs with Auxiliary Parameters
"""

from .base import MCMCResults
from .base_algorithm import BaseAlgorithm
from .algorithm8 import Algorithm8


class Algorithm2(BaseAlgorithm):
    """
    Algorithm 2: Gibbs sampling with m auxiliary parameters

    This is the limit of Algorithm 8 as m → ∞
    In practice, m = 30 gives a good approximation
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100, m: int = 30) -> MCMCResults:
        """Run Algorithm 2 (which is Algorithm 8 with m=30 by default)"""
        algo8 = Algorithm8(self.model)
        return algo8.run(n_iter, burn_in, m)
