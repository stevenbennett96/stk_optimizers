"""
Defines GFN-xTB optimizers.
"""

import os
import logging
import subprocess as sp
import uuid
import shutil
import re

from .optimizers import Optimizer


logger = logging.getLogger(__name__)


class XTBInvalidSolventError(Exception):
    ...


class XTBOptimizerError(Exception):
    ...


class XTBConvergenceError(XTBOptimizerError):
    ...


def is_valid_xtb_solvent(gfn_version, solvent):
    """
    Check if solvent is valid for the given GFN version.
    Parameters
    ----------
    gfn_version : :class:`int`
        GFN parameterization version. Can be: ``0``, ``1`` or ``2``.
    solvent : :class:`str`
        Solvent being tested [1]_.
    Returns
    -------
    :class:`bool`
        ``True`` if solvent is valid.
    References
    ----------
    .. [1] https://xtb-docs.readthedocs.io/en/latest/gbsa.html
    """
    if gfn_version == 0:
        return False
    elif gfn_version == 1:
        valid_solvents = {
            'acetone', 'acetonitrile', 'benzene',
            'CH2Cl2'.lower(), 'CHCl3'.lower(), 'CS2'.lower(),
            'DMSO'.lower(), 'ether', 'H2O'.lower(),
            'methanol', 'THF'.lower(), 'toluene'
        }
        return solvent in valid_solvents
    elif gfn_version == 2:
        valid_solvents = {
            'acetone', 'acetonitrile', 'CH2Cl2'.lower(),
            'CHCl3'.lower(), 'CS2'.lower(), 'DMF'.lower(),
            'DMSO'.lower(), 'ether', 'H2O'.lower(), 'methanol',
            'n-hexane'.lower(), 'THF'.lower(), 'toluene'
        }
        return solvent in valid_solvents


class XTB(Optimizer):
    """
    Uses GFN-xTB [1]_ to optimize molecules.
    Notes
    -----
    When running :meth:`optimize`, this calculator changes the
    present working directory with :func:`os.chdir`. The original
    working directory will be restored even if an error is raised, so
    unless multi-threading is being used this implementation detail
    should not matter.
    If multi-threading is being used an error could occur if two
    different threads need to know about the current working directory
    as :class:`.XTB` can change it from under them.
    Note that this does not have any impact on multi-processing,
    which should always be safe.
    Furthermore, :meth:`optimize` will check that the
    structure is adequately optimized by checking for negative
    frequencies after a Hessian calculation. `max_runs` can be
    provided to the initializer to set the maximum number of
    optimizations which will be attempted at the given
    `opt_level` to obtain an optimized structure. However, we outline
    in the examples how to iterate over `opt_levels` to increase
    convergence criteria and hopefully obtain an optimized structure.
    The presence of negative frequencies can occur even when the
    optimization has converged based on the given `opt_level`.
    Attributes
    ----------
    incomplete : :class:`set` of :class:`.Molecule`
        A :class:`set` of molecules passed to :meth:`optimize` whose
        optimzation was incomplete.
    Examples
    --------
    Note that for :class:`.ConstructedMolecule` objects constructed by
    ``stk``, :class:`XTB` should usually be used in a
    :class:`.Sequence`. This is because xTB only uses
    xyz coordinates as input and so will not recognize the long bonds
    created during construction. An optimizer which can minimize
    these bonds should be used before :class:`XTB`.
    .. code-block:: python
        import stk
        bb1 = stk.BuildingBlock('NCCNCCN', ['amine'])
        bb2 = stk.BuildingBlock('O=CCCC=O', ['aldehyde'])
        polymer = stk.ConstructedMolecule(
            building_blocks=[bb1, bb2],
            topology_graph=stk.polymer.Linear("AB", [0, 0], 3)
        )
        xtb = stk.Sequence(
            stk.UFF(),
            stk.XTB(xtb_path='/opt/gfnxtb/xtb', unlimited_memory=True)
        )
        xtb.optimize(polymer)
    By default, all optimizations with xTB are performed using the
    ``--ohess`` flag, which forces the calculation of a numerical
    Hessian, thermodynamic properties and vibrational frequencies.
    :meth:`optimize` will check that the structure is appropriately
    optimized (i.e. convergence is obtained and no negative vibrational
    frequencies are present) and continue optimizing a structure (up to
    `max_runs` times) until this is achieved. This loop, by
    default, will be performed at the same `opt_level`. The
    following example shows how a user may optimize structures with
    tigher convergence criteria (i.e. different `opt_level`)
    until the structure is sufficiently optimized. Furthermore, the
    calculation of the Hessian can be turned off using
    `max_runs` to ``1`` and `calculate_hessian` to ``False``.
    .. code-block:: python
        # Use crude optimization with max_runs=1 because this will
        # not achieve optimization and rerunning it is unproductive.
        xtb_crude = stk.XTB(
            xtb_path='/opt/gfnxtb/xtb',
            output_dir='xtb_crude',
            unlimited_memory=True,
            opt_level='crude',
            max_runs=1,
            calculate_hessian=True
        )
        # Use normal optimization with max_runs == 2.
        xtb_normal = stk.XTB(
            xtb_path='/opt/gfnxtb/xtb',
            output_dir='xtb_normal',
            unlimited_memory=True,
            opt_level='normal',
            max_runs=2
        )
        # Use vtight optimization with max_runs == 2, which should
        # achieve sufficient optimization.
        xtb_vtight = stk.XTB(
            xtb_path='/opt/gfnxtb/xtb',
            output_dir='xtb_vtight',
            unlimited_memory=True,
            opt_level='vtight',
            max_runs=2
        )
        optimizers = [xtb_crude, xtb_normal, xtb_vtight]
        for optimizer in optimizers:
            optimizer.optimize(polymer)
            if polymer not in optimizer.incomplete:
                break
    References
    ----------
    .. [1] https://xtb-docs.readthedocs.io/en/latest/setup.html
    """

    def __init__(
        self,
        xtb_path,
        gfn_version=2,
        output_dir=None,
        opt_level='normal',
        max_runs=2,
        calculate_hessian=True,
        num_cores=1,
        electronic_temperature=300,
        solvent=None,
        solvent_grid='normal',
        charge=0,
        num_unpaired_electrons=0,
        unlimited_memory=False,
    ):
        """
        Initialize a :class:`XTB` instance.
        Parameters
        ----------
        xtb_path : :class:`str`
            The path to the xTB executable.
        gfn_version : :class:`int`, optional
            Parameterization of GFN to use in xTB.
            For details see
            https://xtb-docs.readthedocs.io/en/latest/basics.html.
        output_dir : :class:`str`, optional
            The name of the directory into which files generated during
            the optimization are written, if ``None`` then
            :func:`uuid.uuid4` is used.
        opt_level : :class:`str`, optional
            Optimization level to use.
            Can be one of ``'crude'``, ``'sloppy'``, ``'loose'``,
            ``'lax'``, ``'normal'``, ``'tight'``, ``'vtight'``
            or ``'extreme'``.
            For details see
            https://xtb-docs.readthedocs.io/en/latest/optimization.html
            .
        max_runs : :class:`int`, optional
            Maximum number of optimizations to attempt in a row.
        calculate_hessian : :class:`bool`, optional
            Toggle calculation of the hessian and vibrational
            frequencies after optimization. ``True`` is required to
            check that the structure is completely optimized.
            ``False`` will drastically speed up the calculation but
            potentially provide incomplete optimizations and forces
            :attr:`max_runs` to be ``1``.
        num_cores : :class:`int`, optional
            The number of cores xTB should use.
        electronic_temperature : :class:`int`, optional
            Electronic temperature in Kelvin.
        solvent : :class:`str`, optional
            Solvent to use in GBSA implicit solvation method.
            For details see
            https://xtb-docs.readthedocs.io/en/latest/gbsa.html.
        solvent_grid : :class:`str`, optional
            Grid level to use in SASA calculations for GBSA implicit
            solvent.
            Can be one of ``'normal'``, ``'tight'``, ``'verytight'``
            or ``'extreme'``.
            For details see
            https://xtb-docs.readthedocs.io/en/latest/gbsa.html.
        charge : :class:`int`, optional
            Formal molecular charge.
        num_unpaired_electrons : :class:`int`, optional
            Number of unpaired electrons.
        unlimited_memory : :class: `bool`, optional
            If ``True`` :meth:`optimize` will be run without
            constraints on the stack size. If memory issues are
            encountered, this should be ``True``, however this may
            raise issues on clusters.
        """

        if solvent is not None:
            solvent = solvent.lower()
            if gfn_version == 0:
                raise XTBInvalidSolventError(
                    f'No solvent valid for version',
                    f' {gfn_version!r}.'
                )
            if not is_valid_xtb_solvent(gfn_version, solvent):
                raise XTBInvalidSolventError(
                    f'Solvent {solvent!r} is invalid for ',
                    f'version {gfn_version!r}.'
                )

        if not calculate_hessian and max_runs != 1:
            max_runs = 1
            logger.warning(
                'Requested that hessian calculation was skipped '
                'but the number of optimizations requested was '
                'greater than 1. The number of optimizations has been '
                'set to 1.'
            )

        self._xtb_path = xtb_path
        self._gfn_version = str(gfn_version)
        self._output_dir = output_dir
        self._opt_level = opt_level
        self._max_runs = max_runs
        self._calculate_hessian = calculate_hessian
        self._num_cores = str(num_cores)
        self._electronic_temperature = str(electronic_temperature)
        self._solvent = solvent
        self._solvent_grid = solvent_grid
        self._charge = str(charge)
        self._num_unpaired_electrons = str(num_unpaired_electrons)
        self._unlimited_memory = unlimited_memory
        self.incomplete = set()

    def _has_neg_frequencies(self, output_file):
        """
        Check for negative frequencies.
        Parameters
        ----------
        output_file : :class:`str`
            Name of output file with xTB results.
        Returns
        -------
        :class:`bool`
            Returns ``True`` if a negative frequency is present.
        """
        xtbext = XTBExtractor(output_file=output_file)
        # Check for one negative frequency, excluding the first
        # 6 frequencies.
        return any(x < 0 for x in xtbext.frequencies[6:])

    def _is_complete(self, output_file):
        """
        Check if xTB optimization has completed and converged.
        Parameters
        ----------
        output_file : :class:`str`
            Name of xTB output file.
        Returns
        -------
        :class:`bool`
            Returns ``False`` if a negative frequency is present.
        Raises
        -------
        :class:`XTBOptimizerError`
            If the optimization failed.
        :class:`XTBConvergenceError`
            If the optimization did not converge.
        """
        if output_file is None:
            # No simulation has been run.
            return False
        # If convergence is achieved, then .xtboptok should exist.
        if os.path.exists('.xtboptok'):
            # Check for negative frequencies in output file if the
            # hessian was calculated.
            # Return True if there exists at least one.
            if self._calculate_hessian:
                return not self._has_neg_frequencies(output_file)
            else:
                return True
        elif os.path.exists('NOT_CONVERGED'):
            raise XTBConvergenceError('Optimization not converged.')
        else:
            raise XTBOptimizerError('Optimization failed to complete')

    def _run_xtb(self, xyz, out_file):
        """
        Run GFN-xTB.
        Parameters
        ----------
        xyz : :class:`str`
            The name of the input structure ``.xyz`` file.
        out_file : :class:`str`
            The name of output file with xTB results.
        Returns
        -------
        None : :class:`NoneType`
        """

        # Modify the memory limit.
        if self._unlimited_memory:
            memory = 'ulimit -s unlimited ;'
        else:
            memory = ''

        # Set optimization level and type.
        if self._calculate_hessian:
            # Do optimization and check hessian.
            optimization = f'--ohess {self._opt_level}'
        else:
            # Do optimization.
            optimization = f'--opt {self._opt_level}'

        if self._solvent is not None:
            solvent = f'--gbsa {self._solvent} {self._solvent_grid}'
        else:
            solvent = ''

        cmd = (
            f'{memory} {self._xtb_path} {xyz} '
            f'--gfn {self._gfn_version} '
            f'{optimization} --parallel {self._num_cores} '
            f'--etemp {self._electronic_temperature} '
            f'{solvent} --chrg {self._charge} '
            f'--uhf {self._num_unpaired_electrons}'
        )

        with open(out_file, 'w') as f:
            # Note that sp.call will hold the program until completion
            # of the calculation.
            sp.call(
                cmd,
                stdin=sp.PIPE,
                stdout=f,
                stderr=sp.PIPE,
                # Shell is required to run complex arguments.
                shell=True
            )

    def _run_optimizations(self, mol):
        """
        Run loop of optimizations on `mol` using xTB.
        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.
        Returns
        -------
        :class:`bool`
            Returns ``True`` if the calculation is complete and
            ``False`` if the calculation is incomplete.
        """
        for run in range(self._max_runs):
            xyz = f'input_structure_{run+1}.xyz'
            out_file = f'optimization_{run+1}.output'
            mol.write(xyz)
            self._run_xtb(xyz=xyz, out_file=out_file)
            # Check if the optimization is complete.
            coord_file = 'xtbhess.coord'
            coord_exists = os.path.exists(coord_file)
            output_xyz = 'xtbopt.xyz'
            opt_complete = self._is_complete(out_file)
            if not opt_complete:
                if coord_exists:
                    # The calculation is incomplete.
                    # Update mol from xtbhess.coord and continue.
                    mol.update_from_file(coord_file)
                else:
                    # Update mol from xtbopt.xyz.
                    mol.update_from_file(output_xyz)
                    # If the negative frequencies are small, then GFN
                    # may not produce the restart file. If that is the
                    # case, exit optimization loop and warn.
                    self.incomplete.add(mol)
                    logging.warning(
                        f'Small negative frequencies present in {mol}.'
                    )
                    return False
            else:
                # Optimization is complete.
                # Update mol from xtbopt.xyz.
                mol.update_from_file(output_xyz)
                break
        return opt_complete

    def _optimize(self, mol):
        """
        Optimize `mol`.
        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.
        Returns
        -------
        None : :class:`NoneType`
        """

        # Remove mol from self.incomplete if present.
        if mol in self.incomplete:
            self.incomplete.remove(mol)

        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.mkdir(output_dir)
        init_dir = os.getcwd()
        os.chdir(output_dir)

        try:
            complete = self._run_optimizations(mol)
        finally:
            os.chdir(init_dir)

        if not complete:
            self.incomplete.add(mol)
            logging.warning(f'Optimization is incomplete for {mol}.')


class XTBExtractor:
    """
    Extracts properties from GFN-xTB output files.
    Attributes
    ----------
    output_file : :class:`str`
        Output file to extract properties from.
    output_lines : :class:`list` : :class:`str`
        :class:`list` of all lines in as :class:`str` in the output
        file.
    total_energy : :class:`float`
        The total energy in the :attr:`output_file` as
        :class:`float`. The energy is in units of a.u..
    homo_lumo_gap : :class:`float`
        The HOMO-LUMO gap in the :attr:`output_file` as
        :class:`float`. The gap is in units of eV.
    fermi_level : :class:`float`
        The Fermi level in the :attr:`output_file` as
        :class:`float` in units of eV.
    qonly_dipole_moment : :class:`list`
        Components of the Q only dipole moment in units
        of Debye in :class:`list` of the form
        ``[x, y, z]``.
    full_dipole_moment : :class:`list`
        Components of the full dipole moment in units
        of Debye in :class:`list` of the form
        ``[x, y, z, total]``.
    qonly_quadrupole_moment : :class:`list`
        Components of the Q only traceless quadrupole moment in units
        of Debye in :class:`list` of the form
        ``[xx, xy, xy, xz, yz, zz]``.
    qdip_quadrupole_moment : :class:`list`
        Components of the Q+Dip traceless quadrupole moment in units of
        Debye in :class:`list` of the form
        ``[xx, xy, xy, xz, yz, zz]``.
    full_quadrupole_moment : :class:`list`
        Components of the full traceless quadrupole moment in units of
        Debye in :class:`list` of the form
        ``[xx, xy, xy, xz, yz, zz]``.
    homo_lumo_occ : :class:`dict`
        :class:`dict` of :class:`list` containing the orbital number,
        energy in eV and occupation of the HOMO and LUMO orbitals in
        the :attr:`output_file`.
    total_free_energy : :class:`float`
        The total free energy in the :attr:`output_file` as
        :class:`float`. The free energy is in units of a.u. and
        calculated at 298.15K.
    frequencies : :class:`list`
        :class:`list` of the vibrational frequencies as :class:`float`
        in the :attr:`output_file`. Vibrational frequencies are in
        units of wavenumber and calculated at 298.15K.
    """

    def __init__(self, output_file):
        """
        Initializes :class:`XTBExtractor`
        Parameters
        ----------
        output_file : :class:`str`
            Output file to extract properties from.
        """
        self.output_file = output_file
        # Explictly set encoding to UTF-8 because default encoding on
        # Windows will fail to read the file otherwise.
        with open(self.output_file, 'r', encoding='UTF-8') as f:
            self.output_lines = f.readlines()

        self._extract_values()

    def _extract_values(self):
        """
        Extract all properties from xTB output file.
        Returns
        -------
        None : :class:`NoneType`
        """

        for i, line in enumerate(self.output_lines):
            if self._check_line(line, 'total_energy'):
                self._extract_total_energy(line)
            elif self._check_line(line, 'homo_lumo_gap'):
                self._extract_homo_lumo_gap(line)
            elif self._check_line(line, 'fermi_level'):
                self._extract_fermi_level(line)
            elif self._check_line(line, 'dipole_moment'):
                self._extract_qonly_dipole_moment(i)
                self._extract_full_dipole_moment(i)
            elif self._check_line(line, 'quadrupole_moment'):
                self._extract_qonly_quadrupole_moment(i)
                self._extract_qdip_quadrupole_moment(i)
                self._extract_full_quadrupole_moment(i)
            elif self._check_line(line, 'homo_lumo_occ_HOMO'):
                self.homo_lumo_occ = {}
                self._extract_homo_lumo_occ(line, 'HOMO')
            elif self._check_line(line, 'homo_lumo_occ_LUMO'):
                self._extract_homo_lumo_occ(line, 'LUMO')
            elif self._check_line(line, 'total_free_energy'):
                self._extract_total_free_energy(line)

        # Frequency formatting requires loop through full file.
        self._extract_frequencies()

    def _check_line(self, line, option):
        """
        Checks a line for a string based on option.
        All formatting based on the 190418 version of xTB.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to check.
        option : :class:`str`
            Define which property and string being checked for.
            Can be one of ``'total_energy'``, ``'homo_lumo_gap'``,
            ``'fermi_level'``, ``'dipole_moment'``,
            ``'quadrupole_moment'``, ``'homo_lumo_occ_HOMO'``,
            ``'homo_lumo_occ_LUMO'``,
            ``'total_free_energy'``.
        Returns
        -------
        :class:`bool`
            Returns ``True`` if the desired string is present.
        """
        options = {
            'total_energy': '          | TOTAL ENERGY  ',
            'homo_lumo_gap': '          | HOMO-LUMO GAP   ',
            'fermi_level': '             Fermi-level        ',
            'dipole_moment': 'molecular dipole:',
            'quadrupole_moment': 'molecular quadrupole (traceless):',
            'homo_lumo_occ_HOMO': '(HOMO)',
            'homo_lumo_occ_LUMO': '(LUMO)',
            'total_free_energy': '          | TOTAL FREE ENERGY  ',
        }

        if options[option] in line:
            return True

    def _extract_total_energy(self, line):
        """
        Updates :attr:`total_energy`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        Returns
        -------
        None : :class:`NoneType`
        """

        nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        string = nums.search(line.rstrip()).group(0)
        self.total_energy = float(string)

    def _extract_homo_lumo_gap(self, line):
        """
        Updates :attr:`homo_lumo_gap`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        Returns
        -------
        None : :class:`NoneType`
        """

        nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        string = nums.search(line.rstrip()).group(0)
        self.homo_lumo_gap = float(string)

    def _extract_fermi_level(self, line):
        """
        Updates :attr:`fermi_level`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        Returns
        -------
        None : :class:`NoneType`
        """

        nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        part2 = line.split('Eh')
        string = nums.search(part2[1].rstrip()).group(0)
        self.fermi_level = float(string)

    def _extract_qonly_dipole_moment(self, index):
        """
        Updates :attr:`qonly_dipole_moment`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        index : :class:`int`
            Index of line in :attr:`output_lines`.
        Returns
        -------
        None : :class:`NoneType`
        """

        sample_set = self.output_lines[index+2].rstrip()

        if 'q only:' in sample_set:
            self.qonly_dipole_moment = [
                float(i)
                for i in sample_set.split(':')[1].split(' ') if i
            ]

    def _extract_full_dipole_moment(self, index):
        """
        Updates :attr:`full_dipole_moment`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        index : :class:`int`
            Index of line in :attr:`output_lines`.
        Returns
        -------
        None : :class:`NoneType`
        """

        sample_set = self.output_lines[index+3].rstrip()

        if 'full:' in sample_set:
            self.full_dipole_moment = [
                float(i)
                for i in sample_set.split(':')[1].split(' ') if i
            ]

    def _extract_qonly_quadrupole_moment(self, index):
        """
        Updates :attr:`qonly_quadrupole_moment`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        index : :class:`int`
            Index of line in :attr:`output_lines`.
        Returns
        -------
        None : :class:`NoneType`
        """

        sample_set = self.output_lines[index+2].rstrip()

        if 'q only:' in sample_set:
            self.qonly_quadrupole_moment = [
                float(i)
                for i in sample_set.split(':')[1].split(' ') if i
            ]

    def _extract_qdip_quadrupole_moment(self, index):
        """
        Updates :attr:`qdip_quadrupole_moment`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        index : :class:`int`
            Index of line in :attr:`output_lines`.
        Returns
        -------
        None : :class:`NoneType`
        """

        sample_set = self.output_lines[index+3].rstrip()

        if 'q+dip:' in sample_set:
            self.qdip_quadrupole_moment = [
                float(i)
                for i in sample_set.split(':')[1].split(' ') if i
            ]

    def _extract_full_quadrupole_moment(self, index):
        """
        Updates :attr:`full_quadrupole_moment`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        index : :class:`int`
            Index of line in :attr:`output_lines`.
        Returns
        -------
        None : :class:`NoneType`
        """

        sample_set = self.output_lines[index+4].rstrip()

        if 'full:' in sample_set:
            self.full_quadrupole_moment = [
                float(i)
                for i in sample_set.split(':')[1].split(' ') if i
            ]

    def _extract_homo_lumo_occ(self, line, orbital):
        """
        Updates :attr:`homo_lumo_occ`.
        Parameters
        ----------
        line : :class:`str`
            Line of output file to extract property from.
        orbital : :class:`str`
            Can be 'HOMO' or 'LUMO'.
        Returns
        -------
        None : :class:`NoneType`
        """

        if orbital == 'HOMO':
            split_line = [i for i in line.rstrip().split(' ') if i]
            # The line is:
            #   Number, occupation, energy (Ha), energy (ev), label
            # Extract:
            #   Number, occupation, energy (eV)
            orbital_val = [
                int(split_line[0]),
                float(split_line[1]),
                float(split_line[3])
            ]
        elif orbital == 'LUMO':
            split_line = [i for i in line.rstrip().split(' ') if i]
            # The line is:
            #   Number, energy (Ha), energy (ev), label
            # Extract:
            #   Number, occupation (zero), energy (eV)
            orbital_val = [
                int(split_line[0]),
                0,
                float(split_line[2])
            ]

        self.homo_lumo_occ[orbital] = orbital_val

    def _extract_total_free_energy(self, line):
        """
        Updates :attr:`total_free_energy`.
        Returns
        -------
        None : :class:`NoneType`
        """

        nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
        string = nums.search(line.rstrip()).group(0)
        self.total_free_energy = float(string)

    def _extract_frequencies(self):
        """
        Updates :attr:`frequencies`.
        Returns
        -------
        None : :class:`NoneType`
        """

        test = '|               Frequency Printout                |'

        # Use a switch to make sure we are extracting values after the
        # final property readout.
        switch = False

        frequencies = []
        for i, line in enumerate(self.output_lines):
            if test in line:
                # Turn on reading as final frequency printout has
                # begun.
                switch = True
            if ' reduced masses (amu)' in line:
                # Turn off reading as frequency section is done.
                switch = False
            if 'eigval :' in line and switch is True:
                samp = line.rstrip().split(':')[1].split(' ')
                split_line = [i for i in samp if i]
                for freq in split_line:
                    frequencies.append(freq)

        self.frequencies = [float(i) for i in frequencies]
