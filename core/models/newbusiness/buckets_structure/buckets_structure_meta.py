from .buckets_structure_mass_model import NewbusinessBucketsMass
from .buckets_structure_priv_model import NewbusinessBucketsPriv
from .buckets_structure_vip_model import NewbusinessBucketsVip


from core.upfm.commons import ModelMetaInfo, ModelContainer

NewbusinessBuckets = ModelContainer(
    models=(NewbusinessBucketsMass, NewbusinessBucketsPriv, NewbusinessBucketsVip)
)
