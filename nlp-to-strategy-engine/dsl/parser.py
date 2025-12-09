"""Trading Strategy DSL Parser using Lark"""
from typing import Optional, Dict, Any, List, Union
from lark import Lark, Transformer, Token
from pathlib import Path
import re
from .ast_nodes import (
    ASTNode, Series, Number, Indicator, TimeReference, 
    Comparison, BooleanOp, Strategy, ASTBuilder
)


class DSLTransformer(Transformer):
    def strategy(self, items):
        entry = items[0]
        exit_rule = items[1] if len(items) > 1 else None
        return ASTBuilder.build_strategy(entry, exit_rule)
    
    def rule_block(self, items):
        return items[0]
    
    def bool_op(self, items):
        left = items[0]
        operator = items[1].value.upper()
        right = items[2]
        return ASTBuilder.build_boolean_op(operator, left, right)
    
    def comparison(self, items):
        left = items[0]
        operator = items[1]
        right = items[2]
        return ASTBuilder.build_comparison(operator, left, right)
    
    def comparison_op(self, items):
        token = items[0]
        return token.value if isinstance(token, Token) else str(token)
    
    def series(self, items):
        name = items[0].value
        return ASTBuilder.build_series(name)
    
    def indicator(self, items):
        name = items[0].value.lower()
        params = items[1]
        return ASTBuilder.build_indicator(name, params)
    
    def indicator_params(self, items):
        params = []
        for item in items:
            if isinstance(item, Token):
                params.append(item.value)
            elif isinstance(item, Number):
                params.append(item.value)
            else:
                params.append(item)
        return params
    
    def time_ref(self, items):
        series = items[0].value
        lag = items[1]
        return ASTBuilder.build_time_ref(series, lag)
    
    def number(self, items):
        value = items[0]
        numeric_str = value.value if isinstance(value, Token) else str(value)
        scale_match = re.match(r'^(-?\d+\.?\d*)([KMB]?)$', numeric_str, re.IGNORECASE)
        if not scale_match:
            raise ValueError(f"Invalid number: {numeric_str}")
        num_part = float(scale_match.group(1))
        scale_part = scale_match.group(2).upper()
        scale_factors = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
        if scale_part:
            num_part *= scale_factors.get(scale_part, 1)
        if num_part == int(num_part):
            num_part = int(num_part)
        return ASTBuilder.build_number(num_part)
    
    def SIGNED_NUMBER(self, token):
        return token


class DSLParser:
    GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
    
    def __init__(self, grammar_text: Optional[str] = None):
        if grammar_text is None:
            if not self.GRAMMAR_PATH.exists():
                raise FileNotFoundError(f"Grammar file not found: {self.GRAMMAR_PATH}")
            with open(self.GRAMMAR_PATH, 'r') as f:
                grammar_text = f.read()
        
        self.grammar = grammar_text
        self.parser = Lark(grammar_text, parser='lalr', transformer=DSLTransformer(),
                          propagate_positions=True, maybe_placeholders=False)
    
    def parse(self, dsl_text: str) -> Strategy:
        try:
            ast = self.parser.parse(dsl_text)
            if not isinstance(ast, Strategy):
                raise ValueError("Parsed result is not a Strategy")
            return ast
        except Exception as e:
            raise Exception(f"DSL Parse Error: {str(e)}")
    
    @staticmethod
    def from_json_rule(rule_dict: Dict[str, Any]) -> Optional[ASTNode]:
        try:
            entry_conditions = []
            for cond in rule_dict.get("entry", []):
                left = DSLParser._build_expr_from_string(cond["left"])
                op = cond["operator"]
                right = DSLParser._build_expr_from_string(cond["right"])
                entry_conditions.append(ASTBuilder.build_comparison(op, left, right))
            
            if not entry_conditions:
                return None
            
            entry = entry_conditions[0]
            for cond in entry_conditions[1:]:
                entry = ASTBuilder.build_boolean_op("AND", entry, cond)
            
            exit_rule = None
            if rule_dict.get("exit"):
                exit_conditions = []
                for cond in rule_dict["exit"]:
                    left = DSLParser._build_expr_from_string(cond["left"])
                    op = cond["operator"]
                    right = DSLParser._build_expr_from_string(cond["right"])
                    exit_conditions.append(ASTBuilder.build_comparison(op, left, right))
                
                exit_rule = exit_conditions[0]
                for cond in exit_conditions[1:]:
                    exit_rule = ASTBuilder.build_boolean_op("AND", exit_rule, cond)
            
            return ASTBuilder.build_strategy(entry, exit_rule)
        except Exception as e:
            raise ValueError(f"Unable to convert rule to AST: {str(e)}")
    
    @staticmethod
    def _build_expr_from_string(expr_str: str) -> ASTNode:
        expr_str = str(expr_str).strip()
        
        try:
            num_val = float(expr_str)
            return ASTBuilder.build_number(int(num_val) if num_val == int(num_val) else num_val)
        except ValueError:
            pass
        
        if "(" in expr_str and ")" in expr_str:
            match = re.match(r'(\w+)\s*\((.*)\)', expr_str)
            if match:
                ind_name = match.group(1).lower()
                params_str = match.group(2).strip()
                params = []
                for param in params_str.split(","):
                    param = param.strip()
                    try:
                        params.append(float(param) if '.' in param else int(param))
                    except ValueError:
                        params.append(param)
                return ASTBuilder.build_indicator(ind_name, params)
        
        if "_" in expr_str or "[" in expr_str:
            if "_" in expr_str:
                parts = expr_str.split("_", 1)
                return ASTBuilder.build_time_ref(parts[0], parts[1])
            elif "[" in expr_str:
                match = re.match(r'(\w+)\[(\d+)\]', expr_str)
                if match:
                    return ASTBuilder.build_time_ref(match.group(1), match.group(2))
        
        return ASTBuilder.build_series(expr_str)


def parse_dsl(dsl_text: str) -> Strategy:
    parser = DSLParser()
    return parser.parse(dsl_text)
