from pydantic import BaseModel
from typing import List
from fastapi import UploadFile, File
from enum import Enum

"""Definitions for the objects used by our resource endpoints."""


class ImagePredictionPayload(BaseModel):
    """
    Pydantic class representing the payload for image classification.
    This class only includes metadata (like the model name) as the image itself
    is uploaded using the FastAPI `UploadFile` mechanism.
    """

    model_name: str

    class Config:
        schema_extra = {
            "example": {
                "model_name": "cat-dog-classifier",
            }
        }


class AnimalType(Enum):
    """
    Enumeration for the possible predicted types of classes in the image.
    """

    CAT = 0
    DOG = 1
    UNKNOWN = 2   # In case the model does the classification 
                # of an image that does not cointain a cat nor a dog

