from .newbusiness_mass_model import NewbusinessMass
from .newbusiness_priv_model import NewbusinessPriv
from .newbusiness_svip_model import NewbusinessSvip
from .newbusiness_bvip_model import NewbusinessBvip

from core.upfm.commons import ModelContainer

Newbusiness = ModelContainer(
    models=(NewbusinessMass, NewbusinessPriv, NewbusinessSvip, NewbusinessBvip)
)
