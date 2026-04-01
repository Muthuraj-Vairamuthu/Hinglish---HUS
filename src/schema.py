"""Data schemas for hallucination benchmarking."""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

TaskType = Literal["factual", "creative", "brainstorm"]
HallucinationLabel = Literal["N", "H-", "H+"]
VariantType = Literal["base", "topic_fronted", "emphasis_shift"]


class PromptItem(BaseModel):
    task_type: TaskType
    prompt_id: str
    variant: VariantType
    prompt: str


class ModelOutput(BaseModel):
    task_type: TaskType
    prompt_id: str
    variant: VariantType
    prompt: str
    model_name: str
    response: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class LabeledExample(BaseModel):
    task_type: TaskType
    prompt_id: str
    variant: VariantType
    prompt: str
    model_name: str
    response: str
    label: HallucinationLabel          # N, H-, H+
    rationale: str
    judge_model: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)