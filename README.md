# NLP-to-Strategy Trading Engine

Convert natural language trading rules into executable backtested strategies using AI-powered parsing.

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Google API Key (for Gemini AI)

### Installation

1. **Clone the repository**
```bash
cd Finacal_Adviser/nlp-to-strategy-engine
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### Run the API Server

```bash
python main.py
```

Server starts at: `http://localhost:8000`
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 Usage

### Example Query

```bash
curl -X POST "http://localhost:8000/api/strategy" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Buy when RSI is above 70 with \$50k and invest 50%. Sell when RSI drops below 30."
```

### Natural Language Examples

```
"Buy when close crosses above 20-day SMA. Sell when RSI drops below 30."

"Start with $100k and invest 25%. Buy when MACD crosses above signal line and volume is above 1M. Exit when RSI exceeds 70."

"Allocate 30% with $75k. Enter when close touches lower Bollinger Band and RSI is below 30. Exit when price reaches upper Bollinger Band."
```

The NLP parser automatically extracts:
- Entry/exit conditions
- Initial capital (e.g., "$50k", "100000 capital")
- Position size (e.g., "50%", "invest 25%")
- Defaults: $10,000 capital, 100% position

## 🧪 Testing

### Run Test Script
```bash
python test_api.py
```

### Test via Browser
1. Navigate to http://localhost:8000/docs
2. Expand `POST /api/strategy`
3. Click "Try it out"
4. Enter your trading rule in the `text` field
5. Execute

### Sample Test Queries

**Simple:**
```
Buy when RSI is above 70. Sell when RSI drops below 30.
```

**Complex:**
```
Start with $100k and invest 25%. Buy when close crosses above 50-day SMA and RSI is between 40 and 60 and volume is above 1.5 million. Exit when close crosses below 20-day EMA or RSI drops below 30.
```

## 📁 Project Structure

```
nlp-to-strategy-engine/
├── main.py                 # FastAPI server
├── nlp_parser/            # Natural language parsing
│   ├── parser.py          # NL → JSON converter
│   ├── schemas.py         # Pydantic models
│   └── utils.py           # LLM client
├── dsl/                   # DSL parser and AST
│   ├── grammar.lark       # Grammar specification
│   ├── parser.py          # DSL parser
│   ├── ast_nodes.py       # AST definitions
│   └── validator.py       # Validation
├── codegen/               # Code generator
│   └── generator.py       # AST → Python
├── backtester/            # Backtest engine
│   ├── engine.py          # Trade simulator
│   └── indicators.py      # Technical indicators
├── data/
│   └── sample_data.csv    # Sample OHLCV data
├── docs/                  # Documentation
│   ├── DSL_SPECIFICATION.md
│   ├── EXAMPLES.md
│   └── ASSUMPTIONS.md
└── requirements.txt
```

## 📖 Documentation

- **DSL Grammar**: [`docs/DSL_SPECIFICATION.md`](https://github.com/AnshulParate2004/Finacal_Adviser/blob/main/nlp-to-strategy-engine/docs/DSL_SPECIFICATION.md)
- **DSL Examples**: [`docs/EXAMPLES.md`](https://github.com/AnshulParate2004/Finacal_Adviser/blob/main/nlp-to-strategy-engine/docs/EXAMPLES.md)
- **Assumptions**: [`docs/ASSUMPTIONS.md`](https://github.com/AnshulParate2004/Finacal_Adviser/blob/main/nlp-to-strategy-engine/docs/ASSUMPTIONS.md)

## 🔧 Pipeline Flow

```
![NLP to Strategy Engine Pipeline](docs/Gemini_Generated_Image_o65mvyo65mvyo65m.png)

Natural Language Input
    ↓
NLP Parser (LLM) → Structured JSON
    ↓
DSL Converter → DSL Code
    ↓
DSL Parser → Abstract Syntax Tree (AST)
    ↓
Validator → Validation
    ↓
Code Generator → Python Function
    ↓
Backtest Engine → Performance Metrics
    ↓
Results
```

## 📈 Supported Indicators

- **Moving Averages**: SMA, EMA
- **Momentum**: RSI, MACD, Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Trend**: ADX
- **Volume**: Volume comparisons

## 🎯 API Endpoints

### `POST /api/strategy`
Main endpoint for processing trading strategies.

**Request:**
```json
{
  "text": "Natural language trading rule"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "input": {
      "original_text": "...",
      "parsed_rule": {...},
      "extracted_capital": 50000,
      "extracted_position_size": 0.5
    },
    "backtest": {
      "total_trades": 5,
      "win_rate": 60.0,
      "total_return_pct": 12.35,
      "sharpe_ratio": 1.45
    }
  }
}
```

### `GET /health`
Health check endpoint.

### `POST /api/check-completeness`
Check if a trading rule is complete before processing.

---

**Repository**: https://github.com/AnshulParate2004/Finacal_Adviser
