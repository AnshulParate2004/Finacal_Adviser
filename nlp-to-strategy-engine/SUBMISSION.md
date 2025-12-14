# NLP-to-Strategy Trading Engine - Submission Summary

## 📦 Repository
**GitHub**: https://github.com/AnshulParate2004/Finacal_Adviser

**Project Path**: `Finacal_Adviser/nlp-to-strategy-engine/`

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd Finacal_Adviser/nlp-to-strategy-engine
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run Server
```bash
python main.py
```
Server: http://localhost:8000
Docs: http://localhost:8000/docs

## 🧪 Testing

### Method 1: Test Script
```bash
python test_api.py
```

### Method 2: cURL
```bash
curl -X POST "http://localhost:8000/api/strategy" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Buy when RSI is above 70 with \$50k and invest 50%. Sell when RSI drops below 30."
```

### Method 3: Interactive Swagger UI
Navigate to http://localhost:8000/docs

## 📚 Documentation

All documentation is in the `docs/` folder:

1. **`DSL_SPECIFICATION.md`** - Complete DSL grammar specification
   - Grammar rules
   - Operators and data types
   - Indicator syntax
   - Validation rules

2. **`EXAMPLES.md`** - DSL usage examples
   - Basic entry/exit
   - Boolean logic (AND/OR)
   - Cross events
   - Time references
   - Complete strategies

3. **`ASSUMPTIONS.md`** - Design decisions and assumptions
   - Data frequency
   - Indicator handling
   - Cross detection logic
   - Time reference behavior

4. **`README.md`** - Quick reference guide

## 🎯 Key Features

### Natural Language Processing
- **Extracts trading rules** from plain English
- **Identifies capital** ($50k, $100,000, etc.)
- **Parses position size** (50%, invest 25%, etc.)
- **Hybrid validation**: Offline regex + LLM fallback

### DSL Engine
- **Lark-based parser** with formal grammar
- **AST generation** with validation
- **Code generation** to executable Python
- **Supports**: 10+ indicators, time references, boolean logic

### Backtesting
- **Trade simulation** on historical data
- **Performance metrics**: Win rate, Sharpe ratio, drawdown
- **Capital management**: Position sizing, P&L tracking

## 📊 Example Workflow

**Input:**
```
"Start with $100k and invest 25%. Buy when close crosses above 50-day SMA and RSI is between 40 and 60. Exit when RSI exceeds 70."
```

**Pipeline:**
```
Natural Language
    ↓ (NLP Parser)
Structured JSON: {entry: [...], exit: [...], capital: 100000, position: 0.25}
    ↓ (DSL Converter)
DSL Code: "ENTRY: close crosses_above sma(close,50) AND rsi(close,14) > 40 AND rsi(close,14) < 60"
    ↓ (Parser)
Abstract Syntax Tree (AST)
    ↓ (Validator)
Validated AST
    ↓ (Code Generator)
Python Trading Function
    ↓ (Backtest Engine)
Results: {trades: 5, win_rate: 60%, return: 12.35%}
```

## 🔧 Technical Stack

- **API**: FastAPI + Uvicorn
- **NLP**: Google Gemini AI (via LangChain)
- **Parser**: Lark parsing library
- **Data**: Pandas + NumPy
- **Indicators**: TA-Lib

## 📁 Project Structure

```
nlp-to-strategy-engine/
├── main.py                 # ⭐ FastAPI server (run this)
├── test_api.py            # ⭐ Test script
├── requirements.txt       # Dependencies
├── .env                   # API key configuration
├── README.md              # ⭐ Main documentation
├── nlp_parser/            # Natural language parsing
├── dsl/                   # DSL parser & AST
├── codegen/               # Code generation
├── backtester/            # Backtest engine
├── data/                  # Sample OHLCV data
└── docs/                  # ⭐ Complete documentation
    ├── DSL_SPECIFICATION.md
    ├── EXAMPLES.md
    ├── ASSUMPTIONS.md
    └── README.md
```

## 🎓 Evaluation Checklist

✅ **GitHub Repository**: Public, accessible  
✅ **README**: Installation, run, test instructions  
✅ **DSL Documentation**: Grammar, examples, assumptions  
✅ **Requirements**: `requirements.txt` included  
✅ **API Server**: FastAPI with interactive docs  
✅ **Tests**: Test script + examples provided  
✅ **Complete Pipeline**: NLP → DSL → AST → Code → Backtest  

## 💡 Sample Test Queries

**Simple:**
```
Buy when RSI is above 70. Sell when RSI drops below 30.
```

**With Capital & Position:**
```
Start with $50k and invest 50%. Buy when close crosses above 20-day SMA. Exit when RSI exceeds 70.
```

**Complex Multi-Indicator:**
```
With $100k allocate 25%. Buy when close crosses above 50-day SMA and RSI is between 40 and 60 and volume is above 1.5 million. Exit when close crosses below 20-day EMA or RSI drops below 30.
```

---

**Contact**: AnshulParate2004 on GitHub  
**Repository**: https://github.com/AnshulParate2004/Finacal_Adviser
