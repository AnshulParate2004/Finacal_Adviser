"""
FastAPI server for NLP-to-Strategy Trading Engine
Converts natural language trading rules to executable strategies with backtesting
"""
import os
import re
import time
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# Import custom modules
from nlp_parser import parse_trading_rule, check_completeness
from nlp_parser.schemas import ParsedStrategy
from dsl import parse_dsl, validate_dsl
from codegen import generate_trading_function
from backtester import BacktestEngine

# Initialize FastAPI app
app = FastAPI(
    title="NLP-to-Strategy Trading Engine",
    description="Convert natural language trading rules to executable backtested strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"


class StrategyResponse(BaseModel):
    """Main strategy processing response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float


# ============================================================================
# Helper Functions
# ============================================================================

def load_sample_data() -> pd.DataFrame:
    """Load sample OHLCV data for backtesting"""
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Sample data not found at {data_path}. "
            "Please ensure data/sample_data.csv exists."
        )
    
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Sample data missing required columns: {missing_cols}")
    
    # Convert date to datetime (handles DD-MM-YYYY and YYYY-MM-DD)
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    df = df.sort_values('date').set_index('date', drop=False)
    
    return df


def _dsl_safe_value(val: Any) -> str:
    """Convert condition value to DSL string; use 'entry_price' not 'entry' to avoid ENTRY keyword."""
    s = str(val).strip()
    # Replace word 'entry' (entry price) with 'entry_price' so parser does not treat it as ENTRY keyword
    if s == "entry" or (s.startswith("entry") and not s.startswith("entry_price")):
        s = s.replace("entry", "entry_price", 1)
    return s


def convert_rule_to_dsl(parsed_strategy: ParsedStrategy) -> str:
    """
    Convert parsed trading rule to DSL format.
    
    Args:
        parsed_strategy: ParsedStrategy object from NLP parser
        
    Returns:
        DSL string representation
    """
    rule = parsed_strategy.rule
    dsl_lines = []
    
    def dsl_cond(c):
        left = _dsl_safe_value(c.left)
        right = _dsl_safe_value(c.right)
        # Normalize "entry" in expressions (e.g. "entry * 0.98") so it parses as entry_price
        left = re.sub(r"\bentry\b", "entry_price", left)
        right = re.sub(r"\bentry\b", "entry_price", right)
        return f"{left} {c.operator} {right}"
    
    # Build entry rule
    if rule.entry:
        entry_conditions = [dsl_cond(c) for c in rule.entry]
        logic_op = " AND " if rule.logic == "AND" else " OR "
        entry_rule = logic_op.join(entry_conditions)
        dsl_lines.append(f"ENTRY: {entry_rule}")
    
    # Build exit rule
    if rule.exit:
        exit_conditions = [dsl_cond(c) for c in rule.exit]
        logic_op = " AND " if rule.logic == "AND" else " OR "
        exit_rule = logic_op.join(exit_conditions)
        dsl_lines.append(f"EXIT: {exit_rule}")
    
    return "\n".join(dsl_lines)


def process_strategy_pipeline(text: str) -> Dict[str, Any]:
    """
    Complete pipeline: NLP → DSL → AST → Code → Backtest
    
    The NLP parser extracts everything from the text:
    - Entry/exit conditions
    - Initial capital (if mentioned, defaults to 10000)
    - Position size (if mentioned, defaults to 1.0)
    
    Args:
        text: Natural language trading rule
        
    Returns:
        Dictionary with complete results
    """
    start_time = time.time()
    
    # Step 1: Parse natural language to structured rule
    # NLP parser extracts capital and position_size from the text automatically
    try:
        parsed_strategy = parse_trading_rule(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"NLP parsing failed: {str(e)}"
        )
    
    # Extract capital and position size from parsed strategy
    initial_capital = parsed_strategy.initial_capital
    position_size = parsed_strategy.position_size
    
    # Step 2: Convert to DSL
    try:
        dsl_code = convert_rule_to_dsl(parsed_strategy)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSL conversion failed: {str(e)}"
        )
    
    # Step 3: Parse DSL to AST
    try:
        ast = parse_dsl(dsl_code)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSL parsing failed: {str(e)}"
        )
    
    # Step 4: Validate DSL/AST
    try:
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise ValueError(
                f"Strategy validation failed: {', '.join(errors)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )
    
    # Step 5: Generate executable trading function
    try:
        trading_function = generate_trading_function(ast)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )
    
    # Step 6: Load data and run backtest
    try:
        df = load_sample_data()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data loading failed: {str(e)}"
        )
    
    try:
        backtester = BacktestEngine(
            initial_capital=initial_capital,
            position_size=position_size
        )
        # OPTIMIZED: Pass pre-generated function to avoid redundant generation
        backtest_result = backtester.run(df, trading_function)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backtest execution failed: {str(e)}"
        )
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Build response
    result = {
        "input": {
            "original_text": text,
            "parsed_rule": parsed_strategy.rule.dict(),
            "indicators_used": parsed_strategy.indicators_used,
            "complexity": parsed_strategy.complexity,
            "dsl_code": dsl_code,
            "ast_tree": ast.to_dict(),
            "extracted_capital": initial_capital,
            "extracted_position_size": position_size
        },
        "signals": {
            "entry_count": len([t for t in backtest_result.trades if t.entry_date]),
            "exit_count": len([t for t in backtest_result.trades if t.exit_date])
        },
        "backtest": {
            "total_trades": backtest_result.total_trades,
            "winning_trades": backtest_result.winning_trades,
            "losing_trades": backtest_result.losing_trades,
            "win_rate": backtest_result.win_rate,
            "total_return_pct": backtest_result.total_return_pct,
            "total_profit": backtest_result.total_profit,
            "max_drawdown": backtest_result.max_drawdown,
            "sharpe_ratio": backtest_result.sharpe_ratio,
            "avg_trade_return": backtest_result.avg_trade_return,
            "max_profit": backtest_result.max_profit,
            "max_loss": backtest_result.max_loss,
            "initial_capital": initial_capital,
            "final_capital": initial_capital + backtest_result.total_profit,
            "position_size": position_size,
            "data_start": str(df['date'].iloc[0]) if not df.empty else None,
            "data_end": str(df['date'].iloc[-1]) if not df.empty else None,
            "data_bars": len(df),
            "trades": [{
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "profit_loss": t.profit,
                "profit_loss_percent": t.return_pct,
                "type": "long"
            } for t in backtest_result.trades]
        },
        "processing_time_ms": processing_time
    }
    
    return result


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NLP-to-Strategy Trading Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "strategy": "POST /api/strategy"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/api/strategy", response_model=StrategyResponse, tags=["Strategy"])
async def process_strategy(
    text: str = Form(..., description="Natural language trading rule with optional capital and position size")
):
    """
    Main endpoint: Convert natural language to strategy and backtest.
    
    The NLP parser automatically extracts from your text:
    - Trading rules (entry/exit conditions)
    - Initial capital (if mentioned, e.g., "$50k", "with 100000 capital")
    - Position size (if mentioned, e.g., "invest 50%", "allocate 25%")
    
    **Example queries:**
    
    Basic (uses defaults: $10,000 capital, 100% position):
    - "Buy when close crosses above 20-day SMA. Sell when RSI drops below 30."
    - "Enter when RSI is above 70. Exit when close crosses below SMA."
    
    With capital specified:
    - "Buy when MACD crosses above signal line with $50,000 capital. Sell when RSI drops below 30."
    - "Start with $100k and buy when close is above SMA(20). Exit when RSI < 30."
    
    With position size specified:
    - "Invest 50% when close crosses above 20-day SMA. Sell when RSI drops below 30."
    - "Allocate 25% when RSI is above 70 with $50k capital. Exit when RSI drops below 30."
    
    **Parameters:**
    - text: Natural language trading rule (required)
      - Can include capital: "$50k", "$100,000", "100k capital", "start with 25000"
      - Can include position: "50%", "invest 25%", "half", "quarter", "allocate 30%"
      - If not mentioned, defaults: $10,000 capital, 100% position
    
    **Returns:**
    - Complete pipeline results including:
      - Parsed rule with extracted capital and position size
      - Entry/exit signals
      - Backtest performance metrics
    """
    try:
        # Process the strategy - NLP parser handles capital/position extraction
        result = process_strategy_pipeline(text)
        
        return StrategyResponse(
            success=True,
            data=result,
            processing_time_ms=result["processing_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return StrategyResponse(
            success=False,
            error=str(e),
            processing_time_ms=0.0
        )


@app.post("/api/check-completeness", tags=["Strategy"])
async def check_rule_completeness(
    text: str = Form(..., description="Natural language text to check")
):
    """
    Check if a trading rule is complete before processing.
    
    Returns whether the rule has all necessary components:
    - Entry action (buy/sell)
    - Complete conditions with indicators
    - Proper comparisons
    
    Note: Capital and position size are optional and will be extracted if present.
    """
    try:
        is_complete, response, used_llm = check_completeness(text)
        
        return {
            "is_complete": is_complete,
            "status": response.status,
            "missing_elements": response.missing_elements,
            "confidence": response.confidence,
            "suggestion": response.suggestion,
            "used_llm": used_llm
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/health", "/api/strategy", "/docs"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again."
        }
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print(" NLP-to-Strategy Trading Engine API")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("\n Ready to process trading strategies!")
    print(" Tip: The NLP parser extracts capital and position size from your text!")
    print("    Example: 'Buy when RSI > 70 with $50k capital investing 50%'")
    print("\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
