from pathlib import Path

from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import data_integrity, train_test_validation

from src.config import REPORTS_DIR, PROCESSED_TEST_IMAGES, PROCESSED_TRAIN_IMAGES, PROCESSED_VALID_IMAGES

#we will validate that our processed data meets a set of requirements. Specifically, rely on pre-defined suites to check the data integrity and ensure the correct split of the data.

train_images_dir = PROCESSED_TRAIN_IMAGES
valid_images_dir = PROCESSED_VALID_IMAGES
test_images_dir = PROCESSED_TEST_IMAGES

train_ds = classification_dataset_from_directory(train_images_dir, object_type="VisionData",image_extension="jpg")
valid_ds = classification_dataset_from_directory(valid_images_dir, object_type="VisionData",image_extension="jpg")
test_ds = classification_dataset_from_directory(test_images_dir, object_type="VisionData",image_extension="jpg")

custom_suite = data_integrity()
custom_suite.add(
    train_test_validation()
)

result = custom_suite.run(train_ds, valid_ds, test_ds)

result.save_as_html(str(REPORTS_DIR / "deepchecks_validation.html"))

print("Deepchecks validation completed and report saved.")