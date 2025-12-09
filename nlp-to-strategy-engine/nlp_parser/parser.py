"""
Hybrid NL parsing logic using offline + LLM checks for completeness.
Reduces API calls from 2 to 1 in most cases.

Flow:
1. Offline Completeness Check (regex/patterns) - FREE
   - If passes → skip LLM completeness check, go directly to parsing
   - If fails → fallback to LLM completeness check
2. LLM Completeness Check (only if offline fails) - $$$
   - Detailed validation for ambiguous rules
3. LLM Parsing (single API call) - $$$
   - Parse complete rule into structured JSON
"""
import re
from typing import Dict, List, Tuple, Optional
from langchain_core.output_parsers import PydanticOutputParser

from .utils import get_llm_client
from .schemas import CompletenessResponse, TradingRule, ParsedStrategy, Condition


class OfflineCompletenessCheck:
    """Offline completeness validation using patterns and regex."""
    
    ENTRY_ACTIONS = {'buy', 'enter', 'long', 'go long', 'bullish', 'open', 'sell', 'short', 'go short', 'bearish'}
    INDICATORS = {'sma', 'ema', 'rsi', 'macd', 'bb', 'bollinger', 'volume', 'close', 'open', 'high', 'low', 'atr', 'adx', 'stochastic', 'obv', 'vwap', 'price'}
    
    @staticmethod
    def check(text: str) -> Tuple[bool, Dict]:
        """Offline completeness check using patterns."""
        details = {'has_entry_action': False, 'has_indicator': False, 'has_complete_condition': False, 'missing_elements': []}
        text_lower = text.lower().strip()
        
        # Check entry action
        has_entry_action = any(action in text_lower for action in OfflineCompletenessCheck.ENTRY_ACTIONS)
        details['has_entry_action'] = has_entry_action
        if not has_entry_action:
            details['missing_elements'].append('Entry action (buy/enter/sell)')
        
        # Check indicator
        has_indicator = any(ind in text_lower for ind in OfflineCompletenessCheck.INDICATORS)
        details['has_indicator'] = has_indicator
        if not has_indicator:
            details['missing_elements'].append('Indicator or price level')
        
        # Check complete comparison
        has_complete_condition = OfflineCompletenessCheck._has_complete_comparison(text_lower)
        details['has_complete_condition'] = has_complete_condition
        if not has_complete_condition:
            details['missing_elements'].append('Complete condition')
        
        is_complete = (has_entry_action and has_indicator and has_complete_condition)
        return is_complete, details
    
    @staticmethod
    def _has_complete_comparison(text: str) -> bool:
        """Check if text has complete comparisons (not dangling)."""
        patterns = [
            r'\b(?:close|open|high|low|volume|price|rsi|sma|ema|macd|bb)\b[^.!?]*?(?:above|below|>|<|>=|<=|==|is|was|crosses).*?(?:\d+|[A-Za-z]+(?:day|week|month|average|high|low|yesterday))',
            r'\b(?:close|open|high|low|volume|rsi|sma|ema|macd|bb)\s*\([^)]*\)[^.!?]*?(?:above|below|>|<|>=|<=|==|is|was|crosses)',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


class NLPParser:
    """Natural Language Parser for trading rules using LLM."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize NLP Parser.
        
        Args:
            model_name: Google Generative AI model name
        """
        self.llm_client = get_llm_client(model_name=model_name, temperature=0.0)
        self.completeness_parser = PydanticOutputParser(pydantic_object=CompletenessResponse)
        self.rule_parser = PydanticOutputParser(pydantic_object=TradingRule)
    
    def check_completeness(self, text: str, use_offline_first: bool = True) -> Tuple[bool, CompletenessResponse, bool]:
        """
        Hybrid completeness check: offline first, then LLM if needed.
        
        Args:
            text: Natural language trading rule text
            use_offline_first: Whether to use offline check first (default: True)
            
        Returns:
            Tuple of (is_complete: bool, response: CompletenessResponse, used_llm: bool)
        """
        # Step 1: Try offline check first (FREE - no API call)
        if use_offline_first:
            is_offline_complete, offline_details = OfflineCompletenessCheck.check(text)
            
            if is_offline_complete:
                # Offline check passed → return immediately, no LLM call needed
                response = CompletenessResponse(
                    is_complete=True,
                    status='complete',
                    missing_elements=[],
                    confidence=0.85,
                    suggestion=None
                )
                return True, response, False  # used_llm=False
        
        # Step 2: Offline check failed or disabled → use LLM ($$$ - API call)
        prompt = f"""
Analyze the following text and determine if it represents a COMPLETE trading rule for ENTRY.

A complete trading rule must have AT MINIMUM:
1. Entry action - explicit buy/enter/long signal
2. Entry condition(s) with clear indicators or price levels
3. Complete condition - no dangling comparisons (e.g., "close is above" without specifying what it's above)

Exit conditions are OPTIONAL. A rule can be complete with only entry conditions.

Text to analyze: "{text}"

Determine if this is a complete trading rule or if it's missing key elements.
If incomplete, identify what's missing and provide a helpful suggestion.

Examples of COMPLETE rules:
- "Buy when close is above 20-day moving average and volume is above 1M" (exit optional)
- "Buy when close is above 20-day moving average and volume is above 1M. Exit when RSI(14) is below 30."
- "Enter when price crosses above yesterday's high."
- "Buy when RSI(14) is above 50 and close is above SMA(20)"

Examples of INCOMPLETE rules:
- "Buy when close is above" (missing the threshold/indicator - what is close above?)
- "When RSI is below 30" (missing entry action - buy or sell?)
- "close is above 20-day moving average" (missing entry action - no buy/sell)
- "above 50" (missing both indicator and action)
"""
        
        try:
            response = self.llm_client.invoke_with_parser(prompt, self.completeness_parser)
            return response.is_complete, response, True  # used_llm=True
        except Exception as e:
            raise Exception(f"Unable to check completeness: {str(e)}. Please try again.")
    
    def parse_rule(self, text: str) -> ParsedStrategy:
        """
        Parse natural language text into structured trading rule.
        
        Hybrid flow:
        1. Check completeness (offline first, LLM fallback)
        2. If incomplete → return missing elements + suggestion, stop here
        3. If complete → parse with LLM (single call)
        
        Args:
            text: Natural language trading rule
            
        Returns:
            ParsedStrategy with structured rule
            
        Raises:
            ValueError: If rule is incomplete or cannot be parsed
        """
        # Step 1: Hybrid completeness check (offline + LLM fallback)
        try:
            is_complete, completeness_response, used_llm = self.check_completeness(text)
        except Exception as e:
            raise ValueError(f"Unable to validate rule: {str(e)}")
        
        # Step 2: If incomplete → return missing elements + suggestion, stop here
        if not is_complete:
            missing = ', '.join(completeness_response.missing_elements)
            suggestion = completeness_response.suggestion or "Please provide a complete trading rule."
            raise ValueError(
                f"Incomplete rule.\n\nMissing: {missing}\n\nSuggestion: {suggestion}"
            )
        
        # Step 3: If complete → parse with LLM only
        prompt = f"""
Convert the following natural language trading rule into a structured JSON format.

Trading Rule: "{text}"

Guidelines:
1. Identify ENTRY conditions (buy/long signals)
2. Identify EXIT conditions (sell/exit signals)
3. For each condition, extract:
   - left: the indicator or variable (e.g., "close", "volume", "sma(close,20)")
   - operator: comparison operator (">", "<", ">=", "<=", "==", "crosses_above", "crosses_below")
   - right: the value or expression to compare against

4. Supported indicators format:
   - sma(field, period) - Simple Moving Average, e.g., sma(close,20)
   - ema(field, period) - Exponential Moving Average, e.g., ema(close,12)
   - rsi(field, period) - Relative Strength Index, e.g., rsi(close,14)
   - macd(field, fast, slow, signal) - MACD, e.g., macd(close,12,26,9)
   - bb(field, period, std) - Bollinger Bands, e.g., bb(close,20,2)
   - volume - Trading volume
   - close, open, high, low - OHLC price data

5. For percentage increases, convert to comparison format:
   - "volume increases by 30%" -> {{"left": "volume", "operator": ">", "right": "volume_prev * 1.3"}}

6. For time-based comparisons:
   - "yesterday's high" -> "high_prev"
   - "last week" -> use appropriate lag notation

7. Multiple conditions connected by "and" should be separate entries in the list.

Examples:

Input: "Buy when close is above 20-day moving average and volume is above 1 million. Exit when RSI(14) is below 30."
Output:
{{
  "entry": [
    {{"left": "close", "operator": ">", "right": "sma(close,20)"}},
    {{"left": "volume", "operator": ">", "right": 1000000}}
  ],
  "exit": [
    {{"left": "rsi(close,14)", "operator": "<", "right": 30}}
  ]
}}

Input: "Enter when price crosses above yesterday's high."
Output:
{{
  "entry": [
    {{"left": "close", "operator": "crosses_above", "right": "high_prev"}}
  ],
  "exit": []
}}
"""
        
        # Step 4: If LLM parse fails → raise error "Unable to parse, please rephrase"
        try:
            trading_rule = self.llm_client.invoke_with_parser(prompt, self.rule_parser)
        except Exception as e:
            raise ValueError(
                f"Unable to parse rule, please rephrase.\n\nError details: {str(e)}"
            )
        
        # Extract indicators used
        indicators_used = self._extract_indicators(trading_rule)
        
        # Determine complexity
        complexity = self._determine_complexity(trading_rule)
        
        parsed_strategy = ParsedStrategy(
            rule=trading_rule,
            original_text=text,
            indicators_used=indicators_used,
            complexity=complexity
        )
        
        return parsed_strategy
    
    def _extract_indicators(self, rule: TradingRule) -> List[str]:
        """
        Extract list of indicators used in the rule.
        
        Args:
            rule: TradingRule object
            
        Returns:
            List of indicator names
        """
        indicators = set()
        
        all_conditions = rule.entry + rule.exit
        
        for condition in all_conditions:
            # Check left side
            left_indicators = re.findall(r'(\w+)\(', str(condition.left))
            indicators.update(left_indicators)
            
            # Check right side
            right_indicators = re.findall(r'(\w+)\(', str(condition.right))
            indicators.update(right_indicators)
        
        return sorted(list(indicators))
    
    def _determine_complexity(self, rule: TradingRule) -> str:
        """
        Determine strategy complexity.
        
        Args:
            rule: TradingRule object
            
        Returns:
            Complexity level: "simple", "medium", or "complex"
        """
        total_conditions = len(rule.entry) + len(rule.exit)
        
        if total_conditions <= 2:
            return "simple"
        elif total_conditions <= 4:
            return "medium"
        else:
            return "complex"


# Convenience functions
def check_completeness(text: str) -> Tuple[bool, CompletenessResponse, bool]:
    """
    Check if text is a complete trading rule.
    
    Args:
        text: Natural language text
        
    Returns:
        Tuple of (is_complete, response, used_llm)
    """
    parser = NLPParser()
    return parser.check_completeness(text)


def parse_trading_rule(text: str) -> ParsedStrategy:
    """
    Parse natural language trading rule into structured format.
    
    Args:
        text: Natural language trading rule
        
    Returns:
        ParsedStrategy object
    """
    parser = NLPParser()
    return parser.parse_rule(text)
