"""
Algorithm 4: Metropolis-Hastings "No Gaps"
"""

from .base import MCMCResults
from .base_algorithm import BaseAlgorithm
from .algorithm3 import Algorithm3


class Algorithm4(BaseAlgorithm):
    """
    Algorithm 4: "No gaps" with Metropolis-Hastings

    This is essentially Algorithm 3 with the "no gaps" structure.
    """

    def run(self, n_iter: int = 1000, burn_in: int = 100) -> MCMCResults:
        """Run Algorithm 4 (which is Algorithm 3)"""
        algo3 = Algorithm3(self.model)
        return algo3.run(n_iter, burn_in)
