# src/stuttgart_reg_agent/schemas.py
from pydantic import BaseModel
from typing import List

class PlotDetails(BaseModel):
    location: str
    size_m2: float
    building_type: str
    floors: int
    height_m: float

class ZoningRecommendation(BaseModel):
    allowed_building_type: str
    max_floors: int
    max_height_m: float
    notes: str

class BuildingCodeCompliance(BaseModel):
    compliant: bool
    violations: List[str]

class AccessibilityAnalysis(BaseModel):
    compliant: bool
    recommendations: List[str]
