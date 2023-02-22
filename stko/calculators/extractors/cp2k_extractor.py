"""
CP2K Extractor
==============

# . :class:`.CP2KExtractor`

Class to extract properties from CP2K output.
"""

import re

from .extractor import Extractor


class CP2KExtractor(Extractor):
    """
    Extracts properties from CP2K output files.

    Attributes
    ----------
    output_file : class:`str`
        Output file to extract properties from.

    """

    def __init__(self, output_file, optimization_step=None):
        """
        Initializes: class: `CP2KExtractor`

        Parameters
        ----------
        output_file: : class: `str`
            Output file to extract properties from.

        optimization_step: : class: `int`, optional
            Step of the optimization to extact energy values from.
            If None, the total energy is extracted from the converged structure.

        """
        self._optimization_step = optimization_step
        super().__init__(output_file)

    def _extract_values(self):
        self._extract_energies(self._optimization_step)

    def _get_properties_dict(self):
        return {
            'total_energy': '  Total energy:        ',
            'dispersion_energy': '  Dispersion energy:  ',
            'exchange_correlation_energy': '  Exchange-correlation energy:  ',
            'core_hamiltonian_energy': '  Core Hamiltonian energy:  ',
        }

    def _extract_energies(self, optimization_step=None):
        """
        Updates :attr: `total_energy`, :attr: `dispersion_energy`, :attr: `exchange_correlation_energy`, and
        :attr: `core_hamiltonian_energy`.

        Parameters
        ----------
        line: : class: `str`
            Line of output file to extract the energy from.

        optimization_step: : class: `int`, optional
            Step of the optimization to extact the energy from.
            If None, the total energy is extracted from the converged structure.

        Returns
        -------
        None: : class: `NoneType`

        """
        if optimization_step is not None:
            test = f' OPTIMIZATION STEP:     {str(optimization_step)}'
        else:
            test = 'GEOMETRY OPTIMIZATION COMPLETED'
        # Swtich ensures that energy is extracted from correct step.
        switch = False
        # Counts the number of properties that have been extracted.
        prop_counter = 0
        properties = list(self._get_properties_dict().values())
        # Filter properties to those containing the term energy.
        properties = list(filter(lambda x: 'energy' in x, properties))
        for line in self.output_lines:
            if test in line:
                switch = True
            elif switch:
                for prop in properties:
                    if self._check_line(line, prop) in line:
                        # Assign class attribute according to the desired property.
                        self.__setattr__(prop, extract_numerical(line))
                        prop_counter += 1
            # Complete when all properties have been extracted.
            if prop_counter == len(properties):
                return


def extract_numerical(self, line):
    """
    Extracts a value from a string.

    Parameters
    ----------
    line: : class: `str`
        Line of output file to extract the value from.
    Returns
    -------
    float: : class: `float`
        Extracted value.

    """
    nums = re.compile(
        r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    string = nums.search(line.rstrip()).group(0)
    return float(string)
