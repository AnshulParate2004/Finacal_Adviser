# NLP-to-Strategy Trading Engine

---
### ✅ When will you make PROFIT?
In an uptrend, strategies that **follow the trend** or **buy the dip** will profit.

**1. Trend Following (The "Best" Strategy)**
*   **Logic:** Buy when price goes UP. Hold while it keeps going UP.
*   **Why it wins:** You ride the main wave of the market.
*   **Example Prompt:**
    > "Buy when Close crosses above SMA(50). Sell when Close crosses below SMA(50)."

**2. Mean Reversion (Buying the Dip)**
*   **Logic:** Buy when price drops suddenly (oversold). Sell when it bounces back.
*   **Why it wins:** In an uptrend, every drop is temporary. Buying low works perfectly.
*   **Example Prompt:**
    > "Buy when RSI is below 30. Sell when RSI is above 70."

---

### ❌ When will you make a LOSS?
To lose money in this market, you have to fight the trend or trade randomly on noise.

**1. Fighting the Trend (The "Worst" Strategy)**
*   **Logic:** Buy when price is skyrocketing (hoping it crashes?) or Sell when it's just starting to rise.
*   **Why it loses:** You are betting against the market momentum.
*   **Example Prompt (Loss Generator):**
    > "Buy when Close crosses below SMA(20). Sell when Close crosses above SMA(20)."
    *   *Analysis:* You buy when it starts falling. You sell when it starts recovering. You capture all the red, none of the green.

**2. Trading on Noise**
*   **Logic:** Using very short-term signals that flicker constantly.
*   **Why it loses:** Fees and "whipsaws" (false signals) eat up your capital.
*   **Example Prompt:**
    > "Buy when Close > Open. Sell when Close < Open."

---
