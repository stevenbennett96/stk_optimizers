from ..base_calculator import Calculator


class Optimizer(Calculator):
    """Abstract base class for optimizers.
    """

    def optimize(self, mol):
        """Optimizes `mol`.

        Parameters
        ----------
        mol : :class:`stk.Molecule`
            The molecule to be optimized.

        Returns
        -------


        """
        ...
