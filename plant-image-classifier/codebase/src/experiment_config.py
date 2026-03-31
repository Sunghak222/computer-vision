from dataclasses import dataclass, asdict
from typing import List, Optional

from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ExperimentConfig:
    model_name: str
    feature_set: List[str]
    img_size: tuple[int, int] = (128, 128)
    augmentation: bool = False
    dataset_tag: str = "default"
    notes: str = ""
    random_seed: int = 42
    model_params: Optional[dict] = None

    def feature_tag(self) -> str:
        return "+".join(self.feature_set)

    def experiment_tag(self) -> str:
        aug_tag = "aug" if self.augmentation else "noaug"
        return f"{self.model_name}__{self.feature_tag()}__{aug_tag}__{self.dataset_tag}"

    def to_dict(self):
        return asdict(self)

def get_feature_ablation_configs(model_name: str = "xgb", augmentation: bool = True):
    return [
        ExperimentConfig(model_name=model_name, feature_set=["hog"], augmentation=augmentation, dataset_tag="base"),
        ExperimentConfig(model_name=model_name, feature_set=["hog", "hsv"], augmentation=augmentation, dataset_tag="base"),
        ExperimentConfig(model_name=model_name, feature_set=["hog", "lbp"], augmentation=augmentation, dataset_tag="base"),
        ExperimentConfig(model_name=model_name, feature_set=["hog", "lbp", "hsv"], augmentation=augmentation, dataset_tag="base"),
        ExperimentConfig(model_name=model_name, feature_set=["hog", "lbp", "hsv", "glcm"], augmentation=augmentation, dataset_tag="base"),
    ]