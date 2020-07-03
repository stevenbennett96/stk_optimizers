"""
Porous Material Calculators
===========================

Contains calculators for porous materials, such as porous organic cages,
metal-organic frameworks.

"""
from ..base_calculator import Calculator
import numpy as np
from ..utilities import (
    atom_vdw_radii,
    get_largest_distance_between_atoms
)
from rdkit.Chem.rdchem import Molecule
import pywindow as pw
import itertools as it


class PorosityProperties(Calculator):
    """
    Contains calculators based on molecular porosity.

    Examples
    --------
    .. code-block:: python

        import stk

        # Create a molecule which we want to determine has a collapsed
        # structure.
        mol1 = stk.BuildingBlock('CCCNCCCN')
        # TODO: Add example cage.
        # Create the energy calculator.
        collapse_calculator = Collapse()

        # Determine whether the cage is collapsed.
        collapsed = collapse_calculator.is_collapsed(cage)

    """

    def is_collapsed(self, mol, conf_id, expected_windows):
        """
        Determines whether `mol` has a collapsed structure.

        Parameters
        ----------
        mol : :class:`stk.Molecule`
            The :class:`stk.Molecule` which we want to determine is
            collapsed.

        conf_id: :class:`int`
            Conformer ID of :mod:`RDKit` molecule to use.
            Only relevent if `mol` is :class:`rdkit.Mol`

        expected_windows: :class:`int` or :class:`NoneType`, optional
            The expected number of windows in the molecule,
            or :class:`NoneType` if the number of windows is unknown.

        Returns
        -------
        :class:`bool` or :class:`NoneType`
            ``True`` if the cage is collapsed, ``False`` if not, or
            :class:`NoneType` if it cannot be identified.

        """
        # Ensure molecule is converted into :mod:`RDKit` format.
        if isinstance(mol, Molecule):
            rdkit_mol = mol.to_rdkit_mol()
        else:
            # This line is for consistent naming of the :mod:`RDKit`
            # `mol`.
            rdkit_mol = mol
        conf = rdkit_mol.GetConformer(conf_id)
        # Reset conformers to ensure molecule only has one.
        rdkit_mol.RemoveAllConformers()
        rdkit_mol.AddConformer(conf)
        window_diff = self.get_window_difference(
            mol=rdkit_mol,
            conf_id=conf_id, expected_windows=expected_windows
        )
        windows = self.get_windows(
            mol=rdkit_mol,
            conf_id=conf_id, expected_windows=expected_windows
        )
        if window_diff is None or windows is None:
            return
        window_std = np.std(windows)
        max_diameter = get_largest_distance_between_atoms(
            mol=rdkit_mol, conf_id=conf_id
        )

    def get_windows(mol, conf_id=-1, expected_windows=None):
        """
        Returns the window diameters for windows in a molecule.

        Parameters
        ----------
        mol: :class:`stk.Molecule`, :class:`rdkit.Mol`
            Molecule to have windows identified.

        conf_id: :class:`int`
            Conformer ID of :mod:`RDKit` molecule to use.
            Only relevent if `mol` is :class:`rdkit.Mol`

        expected_windows: :class:`int` or :class:`NoneType`, optional
            The expected number of windows in the molecule,
            or :class:`NoneType` if the number of windows is unknown.

        Returns
        -------
        all_windows: :class:`np.ndarry` or :class:`NoneType`
            Array of windows diameters, or None if no windows are found,
            or incorrect number of windows found.
        """
        # Ensure molecule is converted into :mod:`RDKit` format.
        if isinstance(mol, Molecule):
            rdkit_mol = mol.to_rdkit_mol()
        else:
            # This line is for consistent naming of the :mod:`RDKit`
            # `mol`.
            rdkit_mol = mol
        conf = rdkit_mol.GetConformer(conf_id)
        # Reset conformers to ensure molecule only has one.
        rdkit_mol.RemoveAllConformers()
        rdkit_mol.AddConformer(conf)
        pw_molecule = pw.molecular.Molecule.load(rdkit_mol)
        all_windows = pw_molecule.calculate_windows(output='windows')
        # Check window calculation was successful.
        if all_windows is None or len(all_windows) != expected_windows:
            return
        return all_windows

    @staticmethod
    def get_window_difference(
            windows):
        """
        Calculates the mean difference in window diameters.

        Parameters
        ----------
        windows: :class:`np.ndarray`
            Array containing window diameters.

        Returns
        -------
        :class:`float`
            Mean difference in window diameters.
        """
        clusters = [list(windows)]
        # Sum the differences in each cluster group,
        # then sum the group totals together.
        diff_sums = []
        for cluster in clusters:
            diff_sum = sum(abs(w1 - w2)
                           for w1, w2 in it.combinations(cluster, 2)
                           )
            diff_num = sum(1 for _ in it.combinations(cluster, 2))
            diff_sums.append(diff_sum / diff_num)
        return np.mean(diff_sums)

    @ staticmethod
    def get_cavity_size(mol, origin, conformer=-1):
        """
        Calculates diameter of the molecule from `origin`.

        The cavity is measured by finding the atom nearest to
        `origin`, correcting for van der Waals diameter of the atom
        and multiplying by -2.

        Parameters
        ----------
        mol: : :class:`stk.Molecule`, :class:`rdkit.Mol`
            Molecule to calculate the cavity size.
        origin:
            Coordinates of the position from which
            the cavity is measured.
        conformer : :class:`int`
            ID of the conformer to use.

        Returns
        -------
        (float): Cavity size of the molecule.
        """
        # Ensure molecule is converted into :mod:`RDKit` format.
        if isinstance(mol, Molecule):
            rdkit_mol = mol.to_rdkit_mol()
        else:
            # This line is for consistent naming of the :mod:`RDKit`
            # `mol`.
            rdkit_mol = mol
        conf = mol.GetConformer(conformer)
        atom_vdw = np.array(
            [atom_vdw_radii[x.GetSymbol()] for x in mol.GetAtoms()]
        )
        distances = euclidean_distances(
            conf.GetPositions(), np.matrix(origin))
        distances = distances.flatten() - atom_vdw
        return -2 * min(distances)
