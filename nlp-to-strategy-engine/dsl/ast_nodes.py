"""AST Node classes for Trading Strategy DSL"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Union, Optional
from enum import Enum


class NodeType(Enum):
    STRATEGY = "strategy"
    BOOL_OP = "boolean_op"
    COMPARISON = "comparison"
    SERIES = "series"
    INDICATOR = "indicator"
    TIME_REF = "time_reference"
    NUMBER = "number"


@dataclass
class ASTNode:
    node_type: str
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Series(ASTNode):
    node_type: str = field(default="series", init=False)
    name: str = ""
    def __post_init__(self):
        if not self.name:
            raise ValueError("Series name cannot be empty")


@dataclass
class Number(ASTNode):
    node_type: str = field(default="number", init=False)
    value: Union[int, float] = 0
    def __post_init__(self):
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Number value must be numeric")


@dataclass
class Indicator(ASTNode):
    node_type: str = field(default="indicator", init=False)
    name: str = ""
    params: List[Union[str, int, float]] = field(default_factory=list)
    def __post_init__(self):
        valid_indicators = {"sma", "ema", "rsi", "macd", "macd_signal", "macd_histogram",
                           "bb_upper", "bb_middle", "bb_lower", "atr", "adx", "stoch", "stoch_d"}
        if self.name.lower() not in valid_indicators:
            raise ValueError(f"Invalid indicator: {self.name}")


@dataclass
class TimeReference(ASTNode):
    node_type: str = field(default="time_reference", init=False)
    series: str = ""
    lag: str = ""
    def __post_init__(self):
        if not self.series or not self.lag:
            raise ValueError("TimeReference requires series and lag")


@dataclass
class Comparison(ASTNode):
    node_type: str = field(default="comparison", init=False)
    operator: str = ""
    left: ASTNode = None
    right: ASTNode = None
    def __post_init__(self):
        valid_ops = {">", "<", ">=", "<=", "==", "!=", "crosses_above", "crosses_below", "crosses", "touches"}
        if self.operator not in valid_ops:
            raise ValueError(f"Invalid operator: {self.operator}")
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.node_type, "operator": self.operator,
                "left": self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
                "right": self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right}


@dataclass
class BooleanOp(ASTNode):
    node_type: str = field(default="boolean_op", init=False)
    operator: str = ""
    left: ASTNode = None
    right: ASTNode = None
    def __post_init__(self):
        if self.operator.upper() not in {"AND", "OR"}:
            raise ValueError(f"Invalid boolean operator: {self.operator}")
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.node_type, "operator": self.operator.upper(),
                "left": self.left.to_dict() if hasattr(self.left, 'to_dict') else self.left,
                "right": self.right.to_dict() if hasattr(self.right, 'to_dict') else self.right}


@dataclass
class Strategy(ASTNode):
    node_type: str = field(default="strategy", init=False)
    entry: Optional[ASTNode] = None
    exit: Optional[ASTNode] = None
    def __post_init__(self):
        if self.entry is None:
            raise ValueError("Strategy requires ENTRY block")
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.node_type,
                "entry": self.entry.to_dict() if hasattr(self.entry, 'to_dict') else self.entry,
                "exit": self.exit.to_dict() if hasattr(self.exit, 'to_dict') else self.exit}


class ASTBuilder:
    @staticmethod
    def build_series(name: str) -> Series:
        return Series(name=name)
    @staticmethod
    def build_number(value: Union[int, float]) -> Number:
        return Number(value=value)
    @staticmethod
    def build_indicator(name: str, params: List) -> Indicator:
        return Indicator(name=name, params=params)
    @staticmethod
    def build_time_ref(series: str, lag: str) -> TimeReference:
        return TimeReference(series=series, lag=lag)
    @staticmethod
    def build_comparison(operator: str, left: ASTNode, right: ASTNode) -> Comparison:
        return Comparison(operator=operator, left=left, right=right)
    @staticmethod
    def build_boolean_op(operator: str, left: ASTNode, right: ASTNode) -> BooleanOp:
        return BooleanOp(operator=operator, left=left, right=right)
    @staticmethod
    def build_strategy(entry: ASTNode, exit: Optional[ASTNode] = None) -> Strategy:
        return Strategy(entry=entry, exit=exit)
