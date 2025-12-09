"""
Pydantic schemas for structured output from LLM parser.
Defines the JSON structure for trading rules.
"""
from typing import List, Union, Optional
from pydantic import BaseModel, Field


class Condition(BaseModel):
    """Represents a single condition in a trading rule."""
    left: str = Field(..., description="Left operand (e.g., 'close', 'volume', 'sma(close,20)')")
    operator: str = Field(..., description="Comparison operator (e.g., '>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below')")
    right: Union[str, int, float] = Field(..., description="Right operand (can be indicator, value, or expression)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "left": "close",
                "operator": ">",
                "right": "sma(close,20)"
            }
        }


class TradingRule(BaseModel):
    """Complete trading rule with entry and exit conditions."""
    entry: List[Condition] = Field(default_factory=list, description="List of entry conditions (AND logic)")
    exit: List[Condition] = Field(default_factory=list, description="List of exit conditions (AND logic)")
    logic: Optional[str] = Field(default="AND", description="Logic operator between conditions (AND/OR)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "entry": [
                    {"left": "close", "operator": ">", "right": "sma(close,20)"},
                    {"left": "volume", "operator": ">", "right": 1000000}
                ],
                "exit": [
                    {"left": "rsi(close,14)", "operator": "<", "right": 30}
                ],
                "logic": "AND"
            }
        }


class CompletenessResponse(BaseModel):
    """Response for completeness check."""
    is_complete: bool = Field(..., description="Whether the rule is complete")
    status: str = Field(..., description="Status: 'complete' or 'incomplete'")
    missing_elements: Optional[List[str]] = Field(default_factory=list, description="List of missing elements if incomplete")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")
    suggestion: Optional[str] = Field(default=None, description="Suggestion to improve the rule if incomplete")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_complete": True,
                "status": "complete",
                "missing_elements": [],
                "confidence": 0.95,
                "suggestion": None
            }
        }


class ParsedStrategy(BaseModel):
    """Complete parsed strategy with metadata."""
    rule: TradingRule = Field(..., description="Parsed trading rule")
    original_text: str = Field(..., description="Original natural language input")
    indicators_used: List[str] = Field(default_factory=list, description="List of indicators referenced")
    complexity: str = Field(default="simple", description="Strategy complexity: simple, medium, complex")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule": {
                    "entry": [
                        {"left": "close", "operator": ">", "right": "sma(close,20)"}
                    ],
                    "exit": [
                        {"left": "rsi(close,14)", "operator": "<", "right": 30}
                    ]
                },
                "original_text": "Buy when close is above 20-day SMA. Exit when RSI(14) is below 30.",
                "indicators_used": ["sma", "rsi"],
                "complexity": "simple"
            }
        }
