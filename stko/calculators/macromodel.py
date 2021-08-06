"""
 MacroModel Calculators
 ======================

 # . :class:`.MacroModelForceFieldEnergy`

 Wrappers for calculators that use the MacroModel software.

 """

import logging
from .calculators import Calculator
from ..packages import MacroModel
from .results import EnergyResults
from uuid import uuid4
from ..utilities import move_generated_macromodel_files
import re

logger = logging.getLogger(__name__)


class MacroModelCalculator(MacroModel, Calculator):
    """
     Base class for MacroModel calculators

     """

    def __init__(
        self,
        macromodel_path,
        output_dir,
        timeout,
        force_field,
    ):
        """
        Initialize a :class:`MacroModelOptimizer` instance.

        Parameters
        ----------
        macromodel_path : :class:`str`
            The full path of the Schrodinger suite within the user's
            machine. For example, on a Linux machine this may be
            something like ``'/opt/schrodinger2017-2'``.

        output_dir : :class:`str`
            The name of the directory into which files generated during
            the optimization are written, if ``None`` then
            :func:`uuid.uuid4` is used.

        timeout : :class:`float`
            The amount in seconds the optimization is allowed to run
            before being terminated. ``None`` means there is no
            timeout.

        force_field : :class:`int`
            The number of the force field to be used.
            Force field arguments can be the following:
            +------------+------------+
            |  1  | MM2               |
            +------------+------------+
            |  2  | MM3               |
            +------------+------------+
            |  3  | AMBER             |
            +------------+------------+
            |  4  | AMBER94           |
            +------------+------------+
            |  5  | OPLSA             |
            +------------+------------+
            |  10 | MMFF94 and MMFF94s|
            +------------+------------+
            |  14 | OPLS_2005         |
            +------------+------------+
            |  16 | OPLS3/3e/4        |
            +------------+------------+
        """

        self._force_field = force_field
        super().__init__(
            macromodel_path=macromodel_path,
            output_dir=output_dir,
            timeout=timeout,
        )


class MacroModellForceFieldEnergy(MacroModelCalculator):
    """
    Uses MacroModel software to calculate energies.

    Examples
    --------
    .. code-block:: python

        import stk
        import stko

        # Create a molecule whose energy
        # we want to calculate.
       mol = stk.BuildingBlock('NCCN')

        # Create the energy calculator.
        opls_ff = stko.MacroModellForceFieldEnergy(
        'path/to/macromodel')

         # Calculate the energy of the molecule.
        results = opls_ff.get_results(mol)
        energy = opls_ff.get_energy(mol)
        unit_string = results.get_unit_string()

    """

    def __init__(
        self,
        macromodel_path,
        output_dir=None,
        timeout=None,
        force_field=16,
    ):
        """
        Initialize a :class:`.MacroModelForceFieldEnergy` instance.

        Parameters
        ----------
        macromodel_path : :class:`str`
             The full path to the MacroModel executable.

        output_dir : :class:`str`, optional
             The path to the output directory.

        timeout : :class:`int`, optional
             The number of seconds to wait for the calculation to
             complete.

        forcefield : :class:`int`, optional
            The force field to use.
            Force field arguments can be the following:
            +------------+------------+
            |  1  | MM2               |
            +------------+------------+
             |  2  | MM3               |
             +------------+------------+
             |  3  | AMBER             |
             +------------+------------+
            |  4  | AMBER94           |
            +------------+------------+
            |  5  | OPLSA             |
            +------------+------------+
            |  10 | MMFF94 and MMFF94s|
            +------------+------------+
            |  14 | OPLS_2005         |
            +------------+------------+
            |  16 | OPLS3/3e/4        |
            +------------+------------+

        Notes
        -----
        The force field argument `16` corresponds to the either
        the OPLS3/3e/4 force field depending on the version of
        MacroModel used.

        """

        super().__init__(
            macromodel_path=macromodel_path,
            output_dir=output_dir,
            timeout=timeout,
            force_field=force_field,
        )

    def get_results(self, mol):
        """
        Calculate the energy of `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to calculate the energy of.

        Returns
        -------
        :class:`.EnergyResults`
            The energy and units of the energy.

         """

        return EnergyResults(
            generator=self.calculate(mol),
            unit_string='kJ mol-1'
        )

    def calculate(self, mol):
        """
        Performs the calculation.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to calculate the energy of.

        """

        run_name = str(uuid4().int)
        if self._output_dir is None:
            output_dir = run_name
        else:
            output_dir = self._output_dir
        mol_path = f'{run_name}.mol'

        mae_path = f'{run_name}.mae'
        # First, write the molecule.
        mol.write(mol_path)
        # MacroModel requires a `.mae` file.
        self._run_structconvert(mol_path, mae_path)
        # Generate the `.com` file.
        self._generate_com(mol, run_name)
        # Run the calculation.
        self._run_bmin(mol, run_name)
        # Convert `.maegz` output to `.mae`.
        self._convert_maegz_to_mae(run_name)
        # Read the results.
        result = self._read_mmo(run_name)
        move_generated_macromodel_files(run_name, output_dir)
        yield result
