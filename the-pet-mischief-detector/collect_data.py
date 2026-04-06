import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# 1. 새 데이터셋 생성
dataset_name = "pet_mischief_dataset"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)

# 2. COCO에서 기본 클래스들 가져오기 (각 100장씩 목표)
print("--- Loading from COCO ---")
coco_classes = ["cat", "dog", "cup", "laptop", "potted plant", "knife", "bottle"]
coco_view = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=coco_classes,
    max_samples=500, # 전체에서 500장 추출
)
dataset.merge_samples(coco_view)

# 3. Open Images에서 COCO에 없는 클래스 추가 (각 100장씩 목표)
print("--- Loading from Open Images ---")
# Open Images는 클래스 이름 대소문자 구분에 주의해야 합니다.
oi_classes = ["Power cable", "Tissue paper", "Bowl"] 
oi_view = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=oi_classes,
    max_samples=300,
)
dataset.merge_samples(oi_view)

# 4. 우리가 정한 클래스 이외의 '사람'이나 '자동차' 등 노이즈 제거
all_my_classes = coco_classes + oi_classes
filtered_view = dataset.filter_labels("ground_truth", F("label").is_in(all_my_classes))

# 5. 결과 확인 및 앱 실행
print(f"Total samples: {len(filtered_view)}")
session = fo.launch_app(filtered_view)
session.wait()