"""
FastAPI Application for NLP-to-Strategy Trading Engine
Single-endpoint API with Server-Sent Events (SSE) support for real-time progress updates
"""
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import asyncio

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


def format_sse(data: dict, event: str = "message") -> str:
    """Format data as Server-Sent Event"""
    json_data = json.dumps(data)
    return f"event: {event}\ndata: {json_data}\n\n"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API information and documentation"""
    return {
        "name": "NLP-to-Strategy Trading Engine API",
        "version": "2.0.0",
        "description": "Convert natural language trading rules to executable strategies and backtest them with SSE support",
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
                "description": "Main endpoint - Complete NL to backtest pipeline (JSON response)",
            },
            "strategy_stream": {
                "path": "/api/strategy/stream",
                "method": "POST",
                "description": "Streaming endpoint - Real-time progress updates via Server-Sent Events",
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
        "version": "2.0.0"
    }


@app.post("/api/strategy")
async def execute_strategy(
    text: str = Form(..., description="Natural language trading rule"),
    initial_capital: Optional[float] = Form(10000.0, description="Initial capital (default: 10000)"),
    position_size: Optional[float] = Form(1.0, description="Position size (default: 1.0)")
):
    """
    **MAIN ENDPOINT** - Complete NLP-to-Strategy Pipeline (Standard JSON Response)
    
    Returns complete results as a single JSON response.
    For real-time progress updates, use /api/strategy/stream instead.
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


@app.post("/api/strategy/stream")
async def execute_strategy_stream(
    text: str = Form(..., description="Natural language trading rule"),
    initial_capital: Optional[float] = Form(10000.0, description="Initial capital (default: 10000)"),
    position_size: Optional[float] = Form(1.0, description="Position size (default: 1.0)")
):
    """
    **STREAMING ENDPOINT** - NLP-to-Strategy Pipeline with Server-Sent Events
    
    Returns real-time progress updates via Server-Sent Events (SSE).
    
    **Event Types:**
    - `progress`: Step-by-step progress updates
    - `complete`: Final results
    - `error`: Error messages
    
    **Example Usage (JavaScript):**
    ```javascript
    const formData = new FormData();
    formData.append('text', 'Buy when RSI is above 70');
    
    const eventSource = new EventSource('/api/strategy/stream');
    
    eventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        console.log(`Step ${data.step}: ${data.message}`);
    });
    
    eventSource.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data);
        console.log('Backtest complete:', data.results);
        eventSource.close();
    });
    
    eventSource.addEventListener('error', (e) => {
        const data = JSON.parse(e.data);
        console.error('Error:', data.error);
        eventSource.close();
    });
    ```
    
    **Example Usage (Python):**
    ```python
    import requests
    
    response = requests.post(
        'http://localhost:8000/api/strategy/stream',
        data={'text': 'Buy when RSI is above 70'},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
    ```
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Validate inputs
            yield format_sse({
                "step": 0,
                "stage": "validation",
                "message": "Validating inputs...",
                "progress": 5
            }, event="progress")
            await asyncio.sleep(0.1)
            
            if not text or not text.strip():
                yield format_sse({
                    "error": "Trading rule text cannot be empty",
                    "code": 400
                }, event="error")
                return
            
            if initial_capital <= 0:
                yield format_sse({
                    "error": "Initial capital must be positive",
                    "code": 400
                }, event="error")
                return
            
            if position_size <= 0 or position_size > 1:
                yield format_sse({
                    "error": "Position size must be between 0 and 1",
                    "code": 400
                }, event="error")
                return
            
            # Step 1: Parse Natural Language → JSON
            yield format_sse({
                "step": 1,
                "stage": "parsing",
                "message": "Parsing natural language rule...",
                "progress": 20
            }, event="progress")
            await asyncio.sleep(0.1)
            
            try:
                parsed_strategy = parse_trading_rule(text)
            except ValueError as e:
                yield format_sse({
                    "error": f"Failed to parse trading rule: {str(e)}",
                    "code": 400
                }, event="error")
                return
            
            yield format_sse({
                "step": 1,
                "stage": "parsing",
                "message": f"✓ Parsed successfully - {len(parsed_strategy.rule.entry)} entry conditions, {len(parsed_strategy.rule.exit)} exit conditions",
                "progress": 30,
                "data": {
                    "indicators": parsed_strategy.indicators_used,
                    "complexity": parsed_strategy.complexity
                }
            }, event="progress")
            await asyncio.sleep(0.1)
            
            # Step 2: Convert JSON → DSL AST
            yield format_sse({
                "step": 2,
                "stage": "ast_building",
                "message": "Building DSL Abstract Syntax Tree...",
                "progress": 40
            }, event="progress")
            await asyncio.sleep(0.1)
            
            try:
                ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
            except Exception as e:
                yield format_sse({
                    "error": f"Failed to build DSL AST: {str(e)}",
                    "code": 400
                }, event="error")
                return
            
            # Step 3: Validate DSL
            yield format_sse({
                "step": 3,
                "stage": "validation",
                "message": "Validating DSL structure...",
                "progress": 50
            }, event="progress")
            await asyncio.sleep(0.1)
            
            is_valid, errors, warnings = validate_dsl(ast)
            if not is_valid:
                yield format_sse({
                    "error": f"Invalid trading rule: {', '.join(errors)}",
                    "code": 400
                }, event="error")
                return
            
            yield format_sse({
                "step": 3,
                "stage": "validation",
                "message": "✓ DSL validated successfully",
                "progress": 55,
                "data": {
                    "warnings": warnings if warnings else []
                }
            }, event="progress")
            await asyncio.sleep(0.1)
            
            # Step 4: Load data
            yield format_sse({
                "step": 4,
                "stage": "data_loading",
                "message": "Loading historical market data...",
                "progress": 60
            }, event="progress")
            await asyncio.sleep(0.1)
            
            try:
                data = load_sample_data()
            except FileNotFoundError as e:
                yield format_sse({
                    "error": str(e),
                    "code": 500
                }, event="error")
                return
            
            yield format_sse({
                "step": 4,
                "stage": "data_loading",
                "message": f"✓ Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}",
                "progress": 65
            }, event="progress")
            await asyncio.sleep(0.1)
            
            # Step 5: Generate signals
            yield format_sse({
                "step": 5,
                "stage": "signal_generation",
                "message": "Generating trading signals...",
                "progress": 70
            }, event="progress")
            await asyncio.sleep(0.1)
            
            try:
                trading_func = generate_trading_function(ast)
                signals = trading_func(data)
            except Exception as e:
                yield format_sse({
                    "error": f"Failed to generate signals: {str(e)}",
                    "code": 500
                }, event="error")
                return
            
            entry_signals = signals['entry']
            exit_signals = signals['exit']
            entry_count = int(entry_signals.sum())
            exit_count = int(exit_signals.sum())
            entry_dates = data.index[entry_signals].strftime('%Y-%m-%d').tolist()
            exit_dates = data.index[exit_signals].strftime('%Y-%m-%d').tolist()
            
            yield format_sse({
                "step": 5,
                "stage": "signal_generation",
                "message": f"✓ Generated {entry_count} entry signals and {exit_count} exit signals",
                "progress": 80,
                "data": {
                    "entry_count": entry_count,
                    "exit_count": exit_count
                }
            }, event="progress")
            await asyncio.sleep(0.1)
            
            # Step 6: Run backtest
            yield format_sse({
                "step": 6,
                "stage": "backtesting",
                "message": "Running backtest simulation...",
                "progress": 85
            }, event="progress")
            await asyncio.sleep(0.1)
            
            try:
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    position_size=position_size
                )
                result = engine.run(data, ast)
            except Exception as e:
                yield format_sse({
                    "error": f"Backtest execution failed: {str(e)}",
                    "code": 500
                }, event="error")
                return
            
            yield format_sse({
                "step": 6,
                "stage": "backtesting",
                "message": f"✓ Backtest complete - {result.total_trades} trades executed",
                "progress": 95
            }, event="progress")
            await asyncio.sleep(0.1)
            
            # Final results
            yield format_sse({
                "step": 7,
                "stage": "complete",
                "message": "Strategy execution complete!",
                "progress": 100,
                "results": {
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
                        "entry_count": entry_count,
                        "exit_count": exit_count,
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
            }, event="complete")
            
        except Exception as e:
            yield format_sse({
                "error": f"Unexpected error: {str(e)}",
                "code": 500
            }, event="error")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
                "strategy": "POST /api/strategy",
                "strategy_stream": "POST /api/strategy/stream"
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
    print("\n🚀 Starting server with SSE support...")
    print("\n📚 Documentation:")
    print("  • Interactive API Docs: http://localhost:8000/docs")
    print("  • ReDoc: http://localhost:8000/redoc")
    print("\n🏥 Health Check:")
    print("  • Health endpoint: http://localhost:8000/health")
    print("\n🎯 Main Endpoints:")
    print("  • POST /api/strategy (Standard JSON response)")
    print("  • POST /api/strategy/stream (Server-Sent Events with real-time progress)")
    print("\n💡 Quick Test (Standard):")
    print('  curl -X POST "http://localhost:8000/api/strategy" \\')
    print('    -F "text=Buy when RSI is above 70. Sell when RSI drops below 30."')
    print("\n💡 Quick Test (SSE Stream):")
    print('  curl -X POST "http://localhost:8000/api/strategy/stream" \\')
    print('    -F "text=Buy when RSI is above 70. Sell when RSI drops below 30."')
    print("\n" + "="*80)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
