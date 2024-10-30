"""
In this script we will use the deepchecks library to validate the data integrity 
of the images in the train, validation, and test datasets, including a custom check for SNR.
"""
from typing import Dict, List

import cv2
import numpy as np
from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import DatasetKind
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.suites import data_integrity, train_test_validation
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
from src.config import (PROCESSED_TEST_IMAGES, PROCESSED_TRAIN_IMAGES,
                        PROCESSED_VALID_IMAGES, REPORTS_DIR)

SHARPNESS_MINIMUM_THRESHOLD = 10.0


def calculate_sharpness(image):
    """
    Calculate the Sharpness Index of an image using the Laplacian.
    """
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()  # pylint: disable=no-member
    return laplacian_var


class SharpnessCheck(TrainTestCheck):
    """Check if the Sharpness Index of images in train and test
    datasets meets the minimum threshold."""

    def initialize_run(self, context: Context):
        """Initialize lists to store Sharpness Index values for train and test datasets."""
        self._sharpness_values = {
            DatasetKind.TRAIN.value: [],
            DatasetKind.TEST.value: [],
        }

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Calculate Sharpness Index for each image in the batch
        and add it to the corresponding list."""
        for image in batch.original_images:
            sharpness_score = calculate_sharpness(image)
            self._sharpness_values[dataset_kind.value].append(sharpness_score)

    def compute(self, context: Context) -> CheckResult:
        """Compute the Sharpness Index results and return them in a CheckResult."""
        sharpness_averages = {
            kind: np.mean(values) if values else 0
            for kind, values in self._sharpness_values.items()
        }

        result_fin = {
            "Average Sharpness": sharpness_averages,
            "Sharpness values per dataset": self._sharpness_values,
        }
        return CheckResult(result_fin)

    def add_sharpness_condition(self, minimum_threshold: float):
        """Add a condition to check that all Sharpness Index values
        are above the minimum threshold."""

        def sharpness_condition(value: Dict[str, List[float]]) -> ConditionResult:
            failed_images = {
                k: [sharpness for sharpness in v if sharpness < minimum_threshold]
                for k, v in value["Sharpness values per dataset"].items()
            }
            if all(len(failed) <= 50 for failed in failed_images.values()):
                return ConditionResult(
                    ConditionCategory.PASS,
                    f"Most of the images have Sharpness Index above {minimum_threshold}.",
                )
            return ConditionResult(
                ConditionCategory.FAIL,
                f"Some images have Sharpness Index below {minimum_threshold}: {failed_images}",
            )

        condition_name = f"Sharpness Index values are >= {minimum_threshold}"
        return self.add_condition(condition_name, sharpness_condition)


train_images_dir = PROCESSED_TRAIN_IMAGES
valid_images_dir = PROCESSED_VALID_IMAGES
test_images_dir = PROCESSED_TEST_IMAGES

train_ds = classification_dataset_from_directory(
    train_images_dir, object_type="VisionData", image_extension="jpg"
)
test_ds = classification_dataset_from_directory(
    test_images_dir, object_type="VisionData", image_extension="jpg"
)

custom_suite = data_integrity()
custom_suite.add(train_test_validation())

sharp_check = SharpnessCheck().add_sharpness_condition(SHARPNESS_MINIMUM_THRESHOLD)
custom_suite.add(sharp_check)

result = custom_suite.run(train_ds, test_ds)
result.save_as_html(str(REPORTS_DIR / "deepchecks_validation.html"))

print("Deepchecks validation completed and report saved.")
