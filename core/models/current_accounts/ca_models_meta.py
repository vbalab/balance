from core.models.current_accounts.general_balance import CurrentAccountsBalance
from core.models.current_accounts.segment_structure import CaSegmentStructure

from core.upfm.commons import ModelMetaInfo, ModelContainer


CurrentAccounts = ModelContainer(models=(CurrentAccountsBalance, CaSegmentStructure))
