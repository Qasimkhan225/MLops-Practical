from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int64
from feast.value_type import ValueType

# Entity
image = Entity(
    name="image_id",
    join_keys=["image_id"],
    value_type=ValueType.INT64
)

# FileSource
digits_source = FileSource(
    name="digits_source",
    path="data/digits_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# 64 pixel features
pixel_fields = [
    Field(name=f"pixel_{r}_{c}", dtype=Float32)
    for r in range(8)
    for c in range(8)
]

# FeatureView
digits_fv = FeatureView(
    name="digit_pixels",
    entities=[image],
    ttl=timedelta(days=365),
    schema=pixel_fields + [Field(name="target", dtype=Int64)],
    source=digits_source,
    online=True
)

# FeatureService
digits_feature_service = FeatureService(
    name="digits_feature_service",
    features=[digits_fv]
)