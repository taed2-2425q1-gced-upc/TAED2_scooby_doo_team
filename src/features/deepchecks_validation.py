from deepchecks.tabular import Dataset
from deepchecks.vision.suites import data_integrity, train_test_validation
from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DATA_DIR, PROCESSED_TEST_IMAGES, PROCESSED_TRAIN_IMAGES, PROCESSED_VALID_IMAGES
import deepchecks.vision as dc_vision

#Load the prepared data
X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")
X_valid = pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")
y_valid = pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")
X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")

#For images, we use Deepchecks' VisionData class
train_images_dir = PROCESSED_TRAIN_IMAGES
valid_images_dir = PROCESSED_VALID_IMAGES
test_images_dir = PROCESSED_TEST_IMAGES

#Load images and labels as a Deepchecks vision dataset
train_ds = dc_vision.classification_dataset_from_directory(
    train_images_dir, object_type="VisionData", label_column=y_train
)
valid_ds = dc_vision.classification_dataset_from_directory(
    valid_images_dir, object_type="VisionData", label_column=y_valid
)
test_ds = dc_vision.classification_dataset_from_directory(
    test_images_dir, object_type="VisionData", label_column=y_test
)

#Create the custom validation suite from Deepchecks
suite = data_integrity()
suite.add(train_test_validation())

#Run the validation suite
result = suite.run(train_ds, valid_ds)

#Save the validation results to an HTML file
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)
result.save_as_html(report_dir / "deepchecks_validation_report.html")

print("Deepchecks validation completed and report saved.")