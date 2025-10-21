from .opt_structure_mass_model import OptStructureMass
from .opt_structure_priv_model import OptStructurePriv
from .opt_structure_svip_model import OptStructureSvip
from .opt_structure_bvip_model import OptStructureBvip

from core.upfm.commons import ModelMetaInfo, ModelContainer

OptStructure = ModelContainer(
    models=(OptStructureMass, OptStructurePriv, OptStructureSvip, OptStructureBvip)
)
