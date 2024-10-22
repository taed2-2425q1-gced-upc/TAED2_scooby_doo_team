"""Definitions for the objects used by our resource endpoints."""

from pydantic import BaseModel, validator
from typing import List
import re

from pydantic import BaseModel, constr

# Modelo para validar el model_name con Pydantic
class PredictionInput(BaseModel):
    model_name: str
