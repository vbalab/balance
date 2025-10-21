import re
from core.upfm.commons import (
    DataLoader,
    ModelTrainer,
    BaseModel,
    ModelMetaInfo,
    ModelContainer,
)

from .meta_model import MetaDataLoader, MetaAdapter, MetaTrainer, CONFIG, gen_file_name


model_meta_info_list = []
for model_params in CONFIG["models_params"]:
    model_meta_info = ModelMetaInfo(
        model_name=gen_file_name(CONFIG, model_params),
        model_trainer=MetaTrainer(model_params, ModelTrainer)(),
        data_loader=MetaDataLoader(model_params, DataLoader)(),
        adapter=MetaAdapter(model_params, BaseModel),
        segment=model_params["segment"][0],
        replenishable_flg=int(re.findall("\d", model_params["optionality"][0])[0]),
        subtraction_flg=int(re.findall("\d", model_params["optionality"][0])[1]),
    )
    model_meta_info_list.append(model_meta_info)

EarlyWithdrawal = ModelContainer(models=tuple(model_meta_info_list))
