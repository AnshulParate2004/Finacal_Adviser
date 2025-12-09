"""DSL Semantic Validator and Quality Checker"""
from typing import List, Dict, Any, Tuple, Optional
from .ast_nodes import (
    ASTNode, Strategy, Series, Number, Indicator, TimeReference,
    Comparison, BooleanOp
)


class ValidationError(Exception):
    pass


class DSLValidator:
    VALID_SERIES = {'close', 'open', 'high', 'low', 'volume'}
    VALID_INDICATORS = {
        'sma': (2, 2), 'ema': (2, 2), 'rsi': (2, 2), 'atr': (1, 1),
        'adx': (1, 1), 'macd': (4, 4), 'macd_signal': (4, 4),
        'macd_histogram': (4, 4), 'bb_upper': (3, 3), 'bb_middle': (3, 3),
        'bb_lower': (3, 3), 'stoch': (4, 4), 'stoch_d': (4, 4),
    }
    VALID_OPERATORS = {">", "<", ">=", "<=", "==", "!=",
                      "crosses_above", "crosses_below", "crosses", "touches"}
    VALID_LAGS = {'prev', '1d_ago', '5d_ago', '1w_ago', '1m_ago'}
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, ast: ASTNode) -> Tuple[bool, List[str], List[str]]:
        self.errors = []
        self.warnings = []
        try:
            if isinstance(ast, Strategy):
                self._validate_strategy(ast)
            else:
                self.errors.append(f"Expected Strategy, got {type(ast).__name__}")
        except Exception as e:
            self.errors.append(f"Validation error: {str(e)}")
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_strategy(self, node: Strategy):
        if node.entry is None:
            self.errors.append("Strategy must have ENTRY block")
            return
        self._validate_node(node.entry)
        if node.exit is not None:
            self._validate_node(node.exit)
    
    def _validate_node(self, node: ASTNode):
        if isinstance(node, Comparison):
            if node.operator not in self.VALID_OPERATORS:
                self.errors.append(f"Invalid operator: {node.operator}")
            self._validate_node(node.left)
            self._validate_node(node.right)
        elif isinstance(node, BooleanOp):
            if node.operator.upper() not in {'AND', 'OR'}:
                self.errors.append(f"Invalid operator: {node.operator}")
            self._validate_node(node.left)
            self._validate_node(node.right)


class DSLHealthCheck:
    @staticmethod
    def check_strategy_quality(ast: Strategy) -> Dict[str, Any]:
        metrics = {
            'entry_complexity': DSLHealthCheck._get_complexity(ast.entry),
            'exit_complexity': DSLHealthCheck._get_complexity(ast.exit) if ast.exit else 0,
            'has_exit': ast.exit is not None,
            'indicator_count': DSLHealthCheck._count_indicators(ast.entry),
            'exit_indicator_count': DSLHealthCheck._count_indicators(ast.exit) if ast.exit else 0,
            'warnings': []
        }
        if metrics['entry_complexity'] > 5:
            metrics['warnings'].append("Entry rule is complex (>5). Consider simplifying.")
        if ast.exit is None:
            metrics['warnings'].append("No EXIT defined. Strategy will hold forever.")
        return metrics
    
    @staticmethod
    def _get_complexity(node: Optional[ASTNode]) -> int:
        if node is None:
            return 0
        if isinstance(node, Comparison):
            return 1
        elif isinstance(node, BooleanOp):
            return DSLHealthCheck._get_complexity(node.left) + DSLHealthCheck._get_complexity(node.right)
        return 0
    
    @staticmethod
    def _count_indicators(node: Optional[ASTNode]) -> int:
        if node is None:
            return 0
        count = 0
        if isinstance(node, Indicator):
            count += 1
        elif isinstance(node, Comparison):
            count += DSLHealthCheck._count_indicators(node.left)
            count += DSLHealthCheck._count_indicators(node.right)
        elif isinstance(node, BooleanOp):
            count += DSLHealthCheck._count_indicators(node.left)
            count += DSLHealthCheck._count_indicators(node.right)
        return count


def validate_dsl(ast: ASTNode) -> Tuple[bool, List[str], List[str]]:
    validator = DSLValidator()
    return validator.validate(ast)


def get_strategy_quality(ast: Strategy) -> Dict[str, Any]:
    return DSLHealthCheck.check_strategy_quality(ast)
