from .maturity_structure_mass_model import MaturityStructureMass
from .maturity_structure_priv_model import MaturityStructurePriv
from .maturity_structure_vip_model import MaturityStructureVip

from core.upfm.commons import ModelContainer

MaturityStructure = ModelContainer(
    model_containers=(
        MaturityStructureMass,
        MaturityStructurePriv,
        MaturityStructureVip,
    )
)
