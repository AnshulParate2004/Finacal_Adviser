"""
FastAPI Application for NLP-to-Strategy Trading Engine
Complete trading strategy pipeline with auto-extracted risk parameters
"""
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
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
    version="2.0.0"
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
        "version": "2.0.0",
        "description": "Convert natural language trading rules to executable strategies and backtest them. Capital and position size are auto-extracted from text!",
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
                    "text": "Natural language trading rule (required). Can include capital like '$50k' or position size like '50%'",
                    "initial_capital": "OPTIONAL override for starting capital (auto-extracted from text if mentioned)",
                    "position_size": "OPTIONAL override for position size (auto-extracted from text if mentioned)"
                },
                "example_curl": 'curl -X POST "http://localhost:8000/api/strategy" -F "text=Buy when RSI is above 70 with $50,000 capital and invest 50%. Sell when RSI drops below 30."',
                "example_python": '''import requests
response = requests.post(
    "http://localhost:8000/api/strategy",
    data={"text": "Buy when RSI is above 70 with $50k. Sell when RSI drops below 30."}
)
print(response.json())'''
            }
        },
        "example_rules": [
            "Buy when close crosses above 20-day SMA with $100k. Sell when close crosses below 20-day SMA.",
            "Buy when RSI is above 70 and volume is above 1 million using 50% position. Sell when RSI drops below 30.",
            "Enter when close is below lower Bollinger Band with $25,000 capital. Exit when close crosses above upper Bollinger Band.",
            "Invest half portfolio when MACD crosses above signal line. Exit when RSI drops below 30."
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "nlp-to-strategy-engine",
        "version": "2.0.0"
    }


@app.post("/api/strategy")
async def execute_strategy(
    text: str = Form(..., description="Natural language trading rule (can include capital/position size)"),
    initial_capital: Optional[float] = Form(None, description="OPTIONAL: Override auto-extracted capital"),
    position_size: Optional[float] = Form(None, description="OPTIONAL: Override auto-extracted position size (0.0-1.0)")
):
    """
    **MAIN ENDPOINT** - Complete NLP-to-Strategy Pipeline
    
    **Auto-Extraction of Risk Parameters:**
    The NLP parser automatically extracts capital and position size from your text!
    
    Examples:
    - "Buy RSI > 70 with $50,000" → auto-extracts $50,000 capital
    - "Buy RSI > 70 using 50%" → auto-extracts 50% position size
    - "Invest half portfolio when..." → auto-extracts 50% position size
    - "Buy with $100k capital and invest 25%..." → auto-extracts both
    
    **Flow:**
    1. Parse natural language → Extract rule + capital + position size
    2. Convert JSON → DSL AST
    3. Validate DSL
    4. Generate trading signals
    5. Run backtest simulation
    6. Return complete results
    
    **Input (Form Data):**
    - `text`: Natural language trading rule (required, can include capital/position mentions)
    - `initial_capital`: OPTIONAL override (if not provided, uses extracted or default $10,000)
    - `position_size`: OPTIONAL override (if not provided, uses extracted or default 100%)
    
    **Example Usage (cURL):**
    ```bash
    # Auto-extract capital and position from text
    curl -X POST "http://localhost:8000/api/strategy" \
      -F "text=Buy when RSI is above 70 with $50k and invest 50%. Sell when RSI drops below 30."
    
    # Override with manual values
    curl -X POST "http://localhost:8000/api/strategy" \
      -F "text=Buy when RSI is above 70. Sell when RSI drops below 30." \
      -F "initial_capital=50000" \
      -F "position_size=0.5"
    ```
    
    **Example Usage (Python):**
    ```python
    import requests
    
    # Auto-extract from text
    response = requests.post(
        "http://localhost:8000/api/strategy",
        data={"text": "Buy when close crosses above 20-day SMA with $100k. Sell when RSI drops below 30."}
    )
    
    result = response.json()
    print(f"Capital used: ${result['data']['config']['initial_capital']}")
    print(f"Total Return: {result['data']['backtest']['total_return_pct']:.2f}%")
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
          "complexity": "medium",
          "extracted_capital": 50000.0,
          "extracted_position_size": 0.5
        },
        "signals": {...},
        "backtest": {...},
        "config": {
          "initial_capital": 50000.0,
          "position_size": 0.5,
          "capital_source": "extracted",
          "position_source": "extracted"
        }
      }
    }
    ```
    """
    try:
        # Validate text input
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Trading rule text cannot be empty")
        
        # Step 1: Parse Natural Language → JSON (auto-extracts capital & position size)
        try:
            parsed_strategy = parse_trading_rule(text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse trading rule: {str(e)}")
        
        # Step 2: Use extracted values OR manual overrides
        # Priority: Manual override > Extracted from text > Default
        final_capital = initial_capital if initial_capital is not None else parsed_strategy.initial_capital
        final_position = position_size if position_size is not None else parsed_strategy.position_size
        
        # Track source of values for transparency
        capital_source = "manual_override" if initial_capital is not None else (
            "extracted" if parsed_strategy.initial_capital != 10000.0 else "default"
        )
        position_source = "manual_override" if position_size is not None else (
            "extracted" if parsed_strategy.position_size != 1.0 else "default"
        )
        
        # Validate final values
        if final_capital <= 0:
            raise HTTPException(status_code=400, detail="Initial capital must be positive")
        
        if final_position <= 0 or final_position > 1:
            raise HTTPException(status_code=400, detail="Position size must be between 0 and 1")
        
        # Step 3: Convert JSON → DSL AST
        try:
            ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to build DSL AST: {str(e)}")
        
        # Step 4: Validate DSL
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid trading rule: {', '.join(errors)}")
        
        # Step 5: Load data and generate signals
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
        
        # Step 6: Run backtest with final capital and position size
        try:
            engine = BacktestEngine(
                initial_capital=final_capital,
                position_size=final_position
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
                    "complexity": parsed_strategy.complexity,
                    "extracted_capital": parsed_strategy.initial_capital,
                    "extracted_position_size": parsed_strategy.position_size
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
                    "initial_capital": final_capital,
                    "position_size": final_position,
                    "capital_source": capital_source,
                    "position_source": position_source
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
    
    print("="*80)
    print("NLP-TO-STRATEGY TRADING ENGINE API v2.0")
    print("="*80)
    print("\n🚀 Starting server with AUTO-EXTRACTION...")
    print("\n✨ NEW: Capital and position size are auto-extracted from text!")
    print("   Examples:")
    print("   • 'Buy with $50k' → auto-extracts $50,000 capital")
    print("   • 'Invest 50%' → auto-extracts 50% position size")
    print("   • 'Use half portfolio' → auto-extracts 50% position")
    print("\n📚 Documentation:")
    print("  • Interactive API Docs: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print("\n🏥 Health Check:")
    print("  • Health endpoint: http://localhost:8000/health")
    print("\n🎯 Main Endpoint:")
    print("  • POST /api/strategy")
    print("\n💡 Quick Test (Auto-extraction):")
    print('  curl -X POST "http://localhost:8000/api/strategy" \\')
    print('    -F "text=Buy when RSI > 70 with $50k and invest 50%. Sell when RSI < 30."')
    print("\n" + "="*80)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
