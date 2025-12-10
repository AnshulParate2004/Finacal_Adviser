"""
FastAPI Application for NLP-to-Strategy Trading Engine
Provides REST API endpoints for the complete trading strategy pipeline
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import io
import json
from datetime import datetime
from pathlib import Path

# Import our modules
from nlp_parser import parse_trading_rule, check_completeness
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
# Request/Response Models
# ============================================================================

class NLPParseRequest(BaseModel):
    """Request model for NLP parsing"""
    text: str = Field(..., description="Natural language trading rule")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Buy when close crosses above 20-day SMA. Sell when RSI drops below 30."
            }
        }


class CompletenessCheckRequest(BaseModel):
    """Request model for completeness check"""
    text: str = Field(..., description="Trading rule text to check")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Buy when close is above"
            }
        }


class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    rule_json: Dict[str, Any] = Field(..., description="Trading rule in JSON format")
    initial_capital: float = Field(10000.0, description="Initial capital for backtest")
    position_size: float = Field(1.0, description="Position size (1.0 = full capital)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rule_json": {
                    "entry": [
                        {"left": "close", "operator": ">", "right": "sma(close,20)"}
                    ],
                    "exit": [
                        {"left": "rsi(close,14)", "operator": "<", "right": 30}
                    ]
                },
                "initial_capital": 10000.0,
                "position_size": 1.0
            }
        }


class EndToEndRequest(BaseModel):
    """Request model for end-to-end pipeline"""
    text: str = Field(..., description="Natural language trading rule")
    initial_capital: Optional[float] = Field(10000.0, description="Initial capital (default: 10000)")
    position_size: Optional[float] = Field(1.0, description="Position size (default: 1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Buy when RSI is above 70. Sell when RSI drops below 30."
            }
        }


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


def dataframe_from_upload(file_content: bytes) -> pd.DataFrame:
    """Convert uploaded CSV to DataFrame"""
    df = pd.read_csv(io.BytesIO(file_content), index_col='date', parse_dates=True)
    
    # Convert to float64
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    return df


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "NLP-to-Strategy Trading Engine API",
        "version": "1.0.0",
        "description": "Convert natural language trading rules to executable strategies",
        "endpoints": {
            "health": "/health",
            "nlp_parse": "/api/nlp/parse",
            "completeness_check": "/api/nlp/check-completeness",
            "validate_rule": "/api/dsl/validate",
            "generate_signals": "/api/signals/generate",
            "backtest": "/api/backtest/run",
            "end_to_end": "/api/pipeline/end-to-end",
            "upload_backtest": "/api/backtest/upload"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "nlp-to-strategy-engine"
    }


@app.post("/api/nlp/parse")
async def nlp_parse(request: NLPParseRequest):
    """
    Parse natural language trading rule into structured JSON
    
    **Flow:**
    1. Check if rule is complete
    2. Parse into structured format
    3. Return JSON rule with metadata
    """
    try:
        parsed_strategy = parse_trading_rule(request.text)
        
        return {
            "success": True,
            "data": {
                "rule": parsed_strategy.rule.dict(),
                "original_text": parsed_strategy.original_text,
                "indicators_used": parsed_strategy.indicators_used,
                "complexity": parsed_strategy.complexity
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")


@app.post("/api/nlp/check-completeness")
async def completeness_check(request: CompletenessCheckRequest):
    """
    Check if trading rule text is complete
    
    **Returns:**
    - is_complete: Boolean
    - missing_elements: List of missing components
    - suggestion: Helpful suggestion to complete the rule
    """
    try:
        is_complete, response, used_llm = check_completeness(request.text)
        
        return {
            "success": True,
            "data": {
                "is_complete": is_complete,
                "status": response.status,
                "missing_elements": response.missing_elements,
                "confidence": response.confidence,
                "suggestion": response.suggestion,
                "used_llm": used_llm
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completeness check error: {str(e)}")


@app.post("/api/dsl/validate")
async def validate_rule(rule_json: Dict[str, Any]):
    """
    Validate trading rule JSON structure
    
    **Checks:**
    - Correct JSON format
    - Valid operators
    - Valid indicators
    - Complete conditions
    """
    try:
        # Parse to AST
        ast = DSLParser.from_json_rule(rule_json)
        
        # Validate
        is_valid, errors, warnings = validate_dsl(ast)
        
        return {
            "success": True,
            "data": {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")


@app.post("/api/signals/generate")
async def generate_signals(rule_json: Dict[str, Any]):
    """
    Generate trading signals from rule (using sample data)
    
    **Returns:**
    - Entry signals count
    - Exit signals count
    - Signal dates
    """
    try:
        # Load sample data
        data = load_sample_data()
        
        # Parse to AST
        ast = DSLParser.from_json_rule(rule_json)
        
        # Validate
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid rule: {errors}")
        
        # Generate signals
        trading_func = generate_trading_function(ast)
        signals = trading_func(data)
        
        # Extract signal dates
        entry_dates = data.index[signals['entry']].strftime('%Y-%m-%d').tolist()
        exit_dates = data.index[signals['exit']].strftime('%Y-%m-%d').tolist()
        
        return {
            "success": True,
            "data": {
                "entry_signals": int(signals['entry'].sum()),
                "exit_signals": int(signals['exit'].sum()),
                "entry_dates": entry_dates,
                "exit_dates": exit_dates,
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
        raise HTTPException(status_code=500, detail=f"Signal generation error: {str(e)}")


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest simulation with sample data
    
    **Returns:**
    - Complete backtest results
    - Performance metrics
    - Trade history
    """
    try:
        # Load sample data
        data = load_sample_data()
        
        # Parse to AST
        ast = DSLParser.from_json_rule(request.rule_json)
        
        # Validate
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid rule: {errors}")
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            position_size=request.position_size
        )
        result = engine.run(data, ast)
        
        return {
            "success": True,
            "data": {
                "backtest_results": result.to_dict(),
                "initial_capital": request.initial_capital,
                "position_size": request.position_size,
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
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@app.post("/api/strategy")
async def simple_strategy(text: str = Form(..., description="Natural language trading rule")):
    """
    **SIMPLEST ENDPOINT** - Just send plain text!
    
    **Usage (Form Data):**
    ```bash
    curl -X POST "http://localhost:8000/api/strategy" \
      -F "text=Buy when RSI is above 70. Sell when RSI drops below 30."
    ```
    
    **Usage (Python):**
    ```python
    import requests
    response = requests.post(
        "http://localhost:8000/api/strategy",
        data={"text": "Buy when RSI is above 70. Sell when RSI drops below 30."}
    )
    ```
    
    Returns complete backtest results with default settings (capital=10000, position=1.0)
    """
    try:
        # Parse NL
        parsed_strategy = parse_trading_rule(text)
        
        # Convert to AST and validate
        ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
        is_valid, errors, warnings = validate_dsl(ast)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid rule: {errors}")
        
        # Load data and generate signals
        data = load_sample_data()
        trading_func = generate_trading_function(ast)
        signals = trading_func(data)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
        result = engine.run(data, ast)
        
        return {
            "success": True,
            "data": {
                "original_text": text,
                "indicators_used": parsed_strategy.indicators_used,
                "signals": {
                    "entry_count": int(signals['entry'].sum()),
                    "exit_count": int(signals['exit'].sum())
                },
                "backtest_results": result.to_dict()
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/pipeline/end-to-end")
async def end_to_end_pipeline(request: EndToEndRequest):
    """
    Complete pipeline: Natural Language → Backtest Results
    
    **Input Options:**
    
    Option 1 - JSON format:
    ```json
    {
      "text": "Buy when RSI is above 70. Sell when RSI drops below 30."
    }
    ```
    
    Option 2 - With parameters:
    ```json
    {
      "text": "Buy when RSI is above 70. Sell when RSI drops below 30.",
      "initial_capital": 50000,
      "position_size": 0.5
    }
    ```
    
    **Flow:**
    1. Parse natural language
    2. Validate rule
    3. Generate signals
    4. Run backtest
    5. Return complete results
    """
    try:
        # Step 1: Parse NL
        parsed_strategy = parse_trading_rule(request.text)
        
        # Step 2: Convert to AST and validate
        ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
        is_valid, errors, warnings = validate_dsl(ast)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid rule: {errors}")
        
        # Step 3: Load data and generate signals
        data = load_sample_data()
        trading_func = generate_trading_function(ast)
        signals = trading_func(data)
        
        # Step 4: Run backtest
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            position_size=request.position_size
        )
        result = engine.run(data, ast)
        
        return {
            "success": True,
            "data": {
                "parsed_rule": {
                    "rule_json": parsed_strategy.rule.dict(),
                    "original_text": parsed_strategy.original_text,
                    "indicators_used": parsed_strategy.indicators_used,
                    "complexity": parsed_strategy.complexity
                },
                "signals": {
                    "entry_count": int(signals['entry'].sum()),
                    "exit_count": int(signals['exit'].sum())
                },
                "backtest_results": result.to_dict(),
                "config": {
                    "initial_capital": request.initial_capital,
                    "position_size": request.position_size
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/api/backtest/upload")
async def backtest_with_upload(
    file: UploadFile = File(..., description="CSV file with OHLCV data"),
    rule_json: str = Form(..., description="Trading rule as JSON string"),
    initial_capital: float = Form(10000.0),
    position_size: float = Form(1.0)
):
    """
    Run backtest with uploaded CSV data
    
    **CSV Format:**
    ```
    date,open,high,low,close,volume
    2023-01-01,100,105,99,103,1000000
    ...
    ```
    """
    try:
        # Parse rule JSON
        rule_dict = json.loads(rule_json)
        
        # Read uploaded file
        content = await file.read()
        data = dataframe_from_upload(content)
        
        # Parse to AST
        ast = DSLParser.from_json_rule(rule_dict)
        
        # Validate
        is_valid, errors, warnings = validate_dsl(ast)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid rule: {errors}")
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=initial_capital,
            position_size=position_size
        )
        result = engine.run(data, ast)
        
        return {
            "success": True,
            "data": {
                "backtest_results": result.to_dict(),
                "initial_capital": initial_capital,
                "position_size": position_size,
                "data_period": {
                    "start": str(data.index[0].date()),
                    "end": str(data.index[-1].date()),
                    "bars": len(data)
                }
            }
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for rule")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload backtest error: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("NLP-TO-STRATEGY TRADING ENGINE API")
    print("="*80)
    print("\nStarting server...")
    print("  • API Documentation: http://localhost:8000/docs")
    print("  • Interactive API: http://localhost:8000/redoc")
    print("  • Health Check: http://localhost:8000/health")
    print("\n" + "="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
