"""
FastAPI Application for NLP-to-Strategy Trading Engine
Single-endpoint API for complete trading strategy pipeline
"""
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import our modules
from nlp_parser import parse_trading_rule
from dsl import DSLParser, validate_dsl
from codegen import generate_trading_function
from backtester import BacktestEngine


# FastAPI app
app = FastAPI(
    title="NLP-to-Strategy Trading Engine API",
    description="Convert natural language trading rules to executable strategies and backtest them",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_sample_data() -> pd.DataFrame:
    """Load sample OHLCV data"""
    data_path = Path(__file__).parent / 'data' / 'sample_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found at {data_path}")
    
    data = pd.read_csv(data_path, index_col='date', parse_dates=True)
    
    # Convert to float64 for TA-Lib
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype('float64')
    
    return data


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information and documentation"""
    return {
        "name": "NLP-to-Strategy Trading Engine API",
        "version": "1.0.0",
        "description": "Convert natural language trading rules to executable strategies and backtest them",
        "documentation": "http://localhost:8000/docs",
        "endpoints": {
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            "strategy": {
                "path": "/api/strategy",
                "method": "POST",
                "description": "Main endpoint - Complete NL to backtest pipeline",
                "parameters": {
                    "text": "Natural language trading rule (required)",
                    "initial_capital": "Starting capital in dollars (optional, default: 10000)",
                    "position_size": "Position size multiplier (optional, default: 1.0)"
                },
                "example_curl": 'curl -X POST "http://localhost:8000/api/strategy" -F "text=Buy when RSI is above 70. Sell when RSI drops below 30."',
                "example_python": '''import requests
response = requests.post(
    "http://localhost:8000/api/strategy",
    data={"text": "Buy when RSI is above 70. Sell when RSI drops below 30."}
)
print(response.json())'''
            }
        },
        "example_rules": [
            "Buy when close crosses above 20-day SMA. Sell when close crosses below 20-day SMA.",
            "Buy when RSI is above 70 and volume is above 1 million. Sell when RSI drops below 30.",
            "Enter when close is below lower Bollinger Band. Exit when close crosses above upper Bollinger Band."
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "nlp-to-strategy-engine",
        "version": "1.0.0"
    }


@app.post("/api/strategy")
async def execute_strategy(
    text: str = Form(..., description="Natural language trading rule"),
    initial_capital: Optional[float] = Form(10000.0, description="Initial capital (default: 10000)"),
    position_size: Optional[float] = Form(1.0, description="Position size (default: 1.0)")
):
    """
    **MAIN ENDPOINT** - Complete NLP-to-Strategy Pipeline
    
    **Flow:**
    1. Parse natural language → Structured JSON
    2. Convert JSON → DSL AST
    3. Validate DSL
    4. Generate trading signals
    5. Run backtest simulation
    6. Return complete results
    
    **Input (Form Data):**
    - `text`: Natural language trading rule (required)
    - `initial_capital`: Starting capital in dollars (optional, default: 10000)
    - `position_size`: Position size multiplier 0.0-1.0 (optional, default: 1.0)
    
    **Example Usage (cURL):**
    ```bash
    curl -X POST "http://localhost:8000/api/strategy" \
      -F "text=Buy when RSI is above 70. Sell when RSI drops below 30." \
      -F "initial_capital=50000" \
      -F "position_size=0.5"
    ```
    
    **Example Usage (Python):**
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/api/strategy",
        data={
            "text": "Buy when close crosses above 20-day SMA. Sell when RSI drops below 30.",
            "initial_capital": 10000,
            "position_size": 1.0
        }
    )
    
    result = response.json()
    print(f"Total Return: {result['data']['backtest']['total_return_pct']:.2f}%")
    print(f"Trades: {result['data']['backtest']['total_trades']}")
    ```
    
    **Returns:**
    ```json
    {
      "success": true,
      "data": {
        "input": {
          "original_text": "...",
          "parsed_rule": {...},
          "indicators_used": ["sma", "rsi"],
          "complexity": "medium"
        },
        "signals": {
          "entry_count": 5,
          "exit_count": 5,
          "entry_dates": ["2023-01-15", ...],
          "exit_dates": ["2023-01-20", ...]
        },
        "backtest": {
          "total_trades": 5,
          "winning_trades": 3,
          "losing_trades": 2,
          "win_rate": 60.0,
          "total_profit": 1234.56,
          "total_return_pct": 12.35,
          "max_drawdown": -5.23,
          "sharpe_ratio": 1.45,
          "trades": [...]
        },
        "config": {
          "initial_capital": 10000.0,
          "position_size": 1.0
        },
        "data_period": {
          "start": "2023-01-01",
          "end": "2023-12-31",
          "bars": 252
        }
      }
    }
    ```
    """
    try:
        # Validate inputs
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Trading rule text cannot be empty")
        
        if initial_capital <= 0:
            raise HTTPException(status_code=400, detail="Initial capital must be positive")
        
        if position_size <= 0 or position_size > 1:
            raise HTTPException(status_code=400, detail="Position size must be between 0 and 1")
        
        # Step 1: Parse Natural Language → JSON
        try:
            parsed_strategy = parse_trading_rule(text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse trading rule: {str(e)}")
        
        # Step 2: Convert JSON → DSL AST
        try:
            ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to build DSL AST: {str(e)}")
        
        # Step 3: Validate DSL
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid trading rule: {', '.join(errors)}")
        
        # Step 4: Load data and generate signals
        try:
            data = load_sample_data()
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        try:
            trading_func = generate_trading_function(ast)
            signals = trading_func(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate signals: {str(e)}")
        
        # Extract signal information
        entry_signals = signals['entry']
        exit_signals = signals['exit']
        entry_dates = data.index[entry_signals].strftime('%Y-%m-%d').tolist()
        exit_dates = data.index[exit_signals].strftime('%Y-%m-%d').tolist()
        
        # Step 5: Run backtest
        try:
            engine = BacktestEngine(
                initial_capital=initial_capital,
                position_size=position_size
            )
            result = engine.run(data, ast)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backtest execution failed: {str(e)}")
        
        # Build response
        return {
            "success": True,
            "data": {
                "input": {
                    "original_text": parsed_strategy.original_text,
                    "parsed_rule": parsed_strategy.rule.dict(),
                    "indicators_used": parsed_strategy.indicators_used,
                    "complexity": parsed_strategy.complexity
                },
                "validation": {
                    "is_valid": is_valid,
                    "warnings": warnings if warnings else []
                },
                "signals": {
                    "entry_count": int(entry_signals.sum()),
                    "exit_count": int(exit_signals.sum()),
                    "entry_dates": entry_dates,
                    "exit_dates": exit_dates
                },
                "backtest": result.to_dict(),
                "config": {
                    "initial_capital": initial_capital,
                    "position_size": position_size
                },
                "data_period": {
                    "start": str(data.index[0].date()),
                    "end": str(data.index[-1].date()),
                    "bars": len(data)
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "available_endpoints": {
                "root": "GET /",
                "health": "GET /health",
                "strategy": "POST /api/strategy"
            }
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check your inputs and try again."
        }
    )


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
