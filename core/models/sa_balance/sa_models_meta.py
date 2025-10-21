from core.models.sa_balance.general_models import (
    SaBalanceMass,
    SaBalancePriv,
    SaBalanceVip,
)
from core.models.sa_balance.product_models import (
    SaKopilkaMass,
    SaKopilkaPriv,
    SaKopilkaVip,
)

from core.upfm.commons import ModelContainer


SaModelsMass = ModelContainer(models=(SaBalanceMass, SaKopilkaMass))

SaModelsPriv = ModelContainer(models=(SaBalancePriv, SaKopilkaPriv))

SaModelsVip = ModelContainer(models=(SaBalanceVip, SaKopilkaVip))


SaModels = ModelContainer(model_containers=(SaModelsMass, SaModelsPriv, SaModelsVip))
