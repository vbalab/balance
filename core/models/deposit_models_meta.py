from core.upfm.commons import ModelMetaInfo, ModelContainer
from core.models.plan_close.plan_close_model import PlanClose
from core.models.newbusiness.maturity_structure.maturity_structure_meta import (
    MaturityStructure,
)
from core.models.newbusiness.opt_structure.opt_structure_meta import OptStructure
from core.models.newbusiness.newbiz.newbusiness_meta import Newbusiness
from core.models.newbusiness.buckets_structure.buckets_structure_meta import (
    NewbusinessBuckets,
)
from core.models.early_withdrawal.early_withdrawal_meta import EarlyWithdrawal
from core.models.renewal.renewal_model import Renewal
from core.models.sa_balance.sa_models_meta import SaModels
from core.models.current_accounts.ca_models_meta import CurrentAccounts

DepositModels = ModelContainer(
    models=(
        PlanClose,
        Renewal,
    ),
    model_containers=(
        NewbusinessBuckets,
        MaturityStructure,
        OptStructure,
        Newbusiness,
        EarlyWithdrawal,
        SaModels,
        CurrentAccounts,
    ),
)
