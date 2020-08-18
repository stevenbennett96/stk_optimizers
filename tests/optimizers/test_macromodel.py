import pytest
import stk
import sys
from stko.optimizers.macromodel import (
    MacroModelMD,
    MacroModelForceField
)

macromodel = pytest.mark.skipif(
    all('macromodel' not in x for x in sys.argv),
    reason="Only run MacroModel tests when explicitly asked.")


@macromodel
def test_restricted_force_field_opt(tmp_cage, macromodel_path):
    ff_opt = MacroModelForceField(macromodel_path, restricted=True)
    opt_mol = ff_opt.optimize(mol)
