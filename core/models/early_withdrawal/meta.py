from upfm.commons import (
    ModelMetaInfo,
    PackageMetaInfo,
)

# from deposit_early_redemption.meta_model import (
#     MetaTrainer,
#     MetaDataLoader,
#     MetaAdapter,
#     CONFIG,
#     gen_file_name,
#     )
import deposit_early_redemption.meta_model
from deposit_early_redemption.meta_model import CONFIG, gen_class_name, gen_file_name

# здесь мы динамически подгружаем созданые динамически классы из meta_model
models = []
for model_params in CONFIG["models_params"]:
    models.append(
        ModelMetaInfo(
            gen_file_name(CONFIG, model_params),
            getattr(
                deposit_early_redemption.meta_model,
                gen_class_name(CONFIG, model_params, "Trainer"),
            ),
            getattr(
                deposit_early_redemption.meta_model,
                gen_class_name(CONFIG, model_params, "DataLoader"),
            ),
            getattr(
                deposit_early_redemption.meta_model,
                gen_class_name(CONFIG, model_params, "Adapter"),
            ),
        )
    )

package_meta_info = PackageMetaInfo(
    models=tuple(models), author="Prokofev Vladimir", email="voprokovef@vtb.ru"
)
