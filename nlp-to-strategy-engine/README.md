# NLP-to-Strategy Trading Engine

A complete pipeline that converts natural language trading rules into executable backtested strategies.

## Overview

This project implements a domain-specific language (DSL) for trading strategies with the following capabilities:
- **Natural Language Parsing**: Convert plain English trading rules to structured format
- **DSL Design**: Clean grammar for expressing entry/exit conditions with technical indicators
- **AST Construction**: Parse DSL into Abstract Syntax Trees
- **Code Generation**: Convert AST to executable Python trading signals
- **Backtesting**: Simulate strategy performance on historical data

## Project Structure

```
nlp-to-strategy-engine/
├── main.py                 # FastAPI server (single-endpoint API)
├── demo/
│   └── run_demo.py        # End-to-end demonstration script
├── nlp_parser/            # Natural language parsing module
│   ├── parser.py          # NL → JSON converter
│   ├── schemas.py         # Pydantic models
│   └── utils.py           # Helper functions
├── dsl/                   # DSL parser and AST builder
│   ├── grammar.lark       # Lark grammar specification
│   ├── parser.py          # DSL parser
│   ├── ast_nodes.py       # AST node definitions
│   └── validator.py       # DSL validation
├── codegen/               # Code generator (AST → Python)
│   └── generator.py       # Signal generation logic
├── backtester/            # Backtest simulation engine
│   ├── engine.py          # Backtest executor
│   └── indicators.py      # Technical indicator helpers
├── data/
│   └── sample_data.csv    # Sample OHLCV data
└── docs/
    └── DSL_Design.md      # Complete DSL specification
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone or extract the project**
   ```bash
   cd nlp-to-strategy-engine
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install manually:
   ```bash
   pip install fastapi uvicorn pandas numpy lark ta-lib pydantic python-multipart
   ```

## Running the System

### Option 1: API Server (Recommended for Testing)

Start the FastAPI server:

```bash
python main.py
```

The server will start at `http://localhost:8000`

**Access interactive documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Test the API:**

Using cURL:
```bash
curl -X POST "http://localhost:8000/api/strategy" \
  -F "text=Buy when RSI is above 70. Sell when RSI drops below 30."
```

Using Python:
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

### Option 2: Command-Line Demo

Run the complete demonstration:

```bash
python demo/run_demo.py
```

This will:
1. Run 3 predefined strategy examples
2. Show the complete pipeline for each
3. Display backtest results

**Interactive Mode:**
Choose option `2` when prompted to enter your own trading rules.

## Example Trading Rules

The system supports natural language rules like:

```
"Buy when close crosses above 20-day SMA. Sell when close crosses below 20-day SMA."

"Buy when RSI is above 70 and volume is above 1 million. Sell when RSI drops below 30."

"Enter when close is below lower Bollinger Band. Exit when close crosses above upper Bollinger Band."

"Buy when MACD crosses above signal line and volume increases by 30%. Sell when RSI drops below 30."
```

## API Endpoints

### Health Check
```
GET /health
```
Returns server health status.

### Main Strategy Endpoint
```
POST /api/strategy
```

**Parameters:**
- `text` (required): Natural language trading rule
- `initial_capital` (optional): Starting capital (default: 10000)
- `position_size` (optional): Position size 0.0-1.0 (default: 1.0)

**Response:**
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
      "exit_count": 5
    },
    "backtest": {
      "total_trades": 5,
      "win_rate": 60.0,
      "total_return_pct": 12.35,
      "max_drawdown": -5.23,
      "sharpe_ratio": 1.45
    }
  }
}
```

## Supported Technical Indicators

- **SMA**: Simple Moving Average - `sma(close, 20)`
- **EMA**: Exponential Moving Average - `ema(close, 12)`
- **RSI**: Relative Strength Index - `rsi(close, 14)`
- **MACD**: Moving Average Convergence Divergence - `macd(close, 12, 26, 9)`
- **Bollinger Bands**: `bb_upper(close, 20, 2)`, `bb_lower(close, 20, 2)`
- **ATR**: Average True Range - `atr(14)`
- **ADX**: Average Directional Index - `adx(14)`
- **Stochastic**: `stoch(close, 14, 3, 3)`

## DSL Specification

See [docs/DSL_Design.md](docs/DSL_Design.md) for complete DSL grammar, examples, and design documentation.

**Key features:**
- Entry/Exit rules
- Boolean logic (AND/OR with parentheses)
- Comparison operators (>, <, >=, <=, ==, !=)
- Cross events (crosses_above, crosses_below)
- Time references (close_prev, high_1d_ago, etc.)
- Technical indicators
- Number formatting (1M = 1,000,000)

## Testing

Run the demo script to verify all components:

```bash
python demo/run_demo.py
```

This tests:
1. ✅ Natural Language Parsing
2. ✅ JSON → DSL Conversion
3. ✅ AST Construction and Validation
4. ✅ Code Generation
5. ✅ Signal Generation
6. ✅ Backtest Execution

## Architecture

### Pipeline Flow

```
Natural Language Input
    ↓
[NLP Parser] → Structured JSON
    ↓
[DSL Parser] → Abstract Syntax Tree (AST)
    ↓
[Validator] → Validation & Error Checking
    ↓
[Code Generator] → Python Trading Function
    ↓
[Signal Generator] → Entry/Exit Signals
    ↓
[Backtest Engine] → Performance Metrics
    ↓
Results Output
```

### Module Responsibilities

- **nlp_parser**: Regex + pattern matching to convert NL to JSON
- **dsl**: Lark-based parser with grammar and AST construction
- **codegen**: AST → executable Python signal generation
- **backtester**: Trade simulation with P&L tracking
- **main.py**: FastAPI server with single unified endpoint

## Performance Metrics

The backtest engine calculates:
- **Total Trades**: Number of complete round-trip trades
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Cumulative percentage return
- **Total Profit**: Absolute profit in dollars
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Average Win/Loss**: Mean profit per winning/losing trade

## Limitations & Assumptions

1. **Daily bars**: System assumes daily OHLCV data
2. **No slippage**: Trades execute at exact close prices
3. **No commissions**: Zero transaction costs
4. **Full position exits**: All positions closed on exit signal
5. **Single asset**: One asset traded at a time
6. **No short selling**: Long-only strategies

## Troubleshooting

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Or install individually
pip install fastapi uvicorn pandas numpy lark ta-lib pydantic
```

### TA-Lib Installation Issues

On Windows:
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

On Linux/Mac:
```bash
# Install TA-Lib C library first
# Ubuntu/Debian
sudo apt-get install ta-lib

# Mac
brew install ta-lib

# Then install Python wrapper
pip install ta-lib
```

### API Not Starting
```bash
# Check if port 8000 is available
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Use different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Future Enhancements

Potential improvements:
- [ ] Multi-asset portfolio support
- [ ] Short selling capabilities
- [ ] Transaction costs and slippage modeling
- [ ] Real-time data integration
- [ ] More sophisticated position sizing
- [ ] Risk management rules (stop-loss, take-profit)
- [ ] Parameter optimization
- [ ] Walk-forward analysis

## License

This project is provided as-is for educational and evaluation purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Built for the NLP → DSL → Strategy Execution Assignment**
