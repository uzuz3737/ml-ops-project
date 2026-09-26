from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., description="Amount of given credit (NT dollar)", example=20000.0)
    SEX: int = Field(..., description="Gender (1=male, 2=female)", ge=1, le=2, example=2)
    EDUCATION: int = Field(..., description="Education (1=grad, 2=univ, 3=high school, 4=other)", ge=0, le=6, example=2)
    MARRIAGE: int = Field(..., description="Marital status (1=married, 2=single, 3=others)", ge=0, le=3, example=1)
    AGE: int = Field(..., description="Age in years", ge=18, le=100, example=24)
    PAY_0: int = Field(..., description="Repayment status in Sept (-1=pay duly, 1=delay 1 mo, etc.)", ge=-2, le=9, example=2)
    PAY_2: int = Field(..., description="Repayment status in Aug", ge=-2, le=9, example=2)
    PAY_3: int = Field(..., description="Repayment status in July", ge=-2, le=9, example=-1)
    PAY_4: int = Field(..., description="Repayment status in June", ge=-2, le=9, example=-1)
    PAY_5: int = Field(..., description="Repayment status in May", ge=-2, le=9, example=-2)
    PAY_6: int = Field(..., description="Repayment status in April", ge=-2, le=9, example=-2)
    BILL_AMT1: float = Field(..., description="Bill statement amount Sept", example=3913.0)
    BILL_AMT2: float = Field(..., description="Bill statement amount Aug", example=3102.0)
    BILL_AMT3: float = Field(..., description="Bill statement amount July", example=689.0)
    BILL_AMT4: float = Field(..., description="Bill statement amount June", example=0.0)
    BILL_AMT5: float = Field(..., description="Bill statement amount May", example=0.0)
    BILL_AMT6: float = Field(..., description="Bill statement amount April", example=0.0)
    PAY_AMT1: float = Field(..., description="Amount paid in Sept", example=0.0)
    PAY_AMT2: float = Field(..., description="Amount paid in Aug", example=689.0)
    PAY_AMT3: float = Field(..., description="Amount paid in July", example=0.0)
    PAY_AMT4: float = Field(..., description="Amount paid in June", example=0.0)
    PAY_AMT5: float = Field(..., description="Amount paid in May", example=0.0)
    PAY_AMT6: float = Field(..., description="Amount paid in April", example=0.0)


class SinglePredictionResponse(BaseModel):
    default_prediction: int = Field(..., description="0 = No default, 1 = Default expected")
    default_probability: float = Field(..., description="Estimated probability of default")
    model_version: str = Field(..., description="Model identifier / version used for scoring")
    prediction_id: Optional[str] = Field(None, description="Identifier used to attach a later ground-truth label")


class BatchCustomerFeatures(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
    count: int
    model_version: str


class PredictionFeedbackRequest(BaseModel):
    prediction_id: str = Field(..., min_length=1)
    actual_default: int = Field(..., ge=0, le=1, description="Observed outcome: 0 = paid, 1 = default")


class PredictionFeedbackResponse(BaseModel):
    status: str
    prediction_id: str
