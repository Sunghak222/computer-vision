import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

fo.close_app()

# 기존 데이터셋 삭제 (이름 확인 후 삭제)
if "pet_mischief_dataset" in fo.list_datasets():
    fo.delete_dataset("pet_mischief_dataset")
    print("기존 데이터셋 삭제 완료!")