"""Quick test script for the API"""
import requests

# Test the API
url = "http://localhost:8000/api/strategy"
data = {
    "text": "Buy when RSI is above 70 with $50k and invest 50%. Sell when RSI drops below 30."
}

print("Testing API with query:")
print(data["text"])
print("\nSending request...")

try:
    response = requests.post(url, data=data)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print(f"\nExtracted Capital: ${result['data']['input']['extracted_capital']:,.2f}")
        print(f"Extracted Position Size: {result['data']['input']['extracted_position_size']*100}%")
        print(f"\nTotal Trades: {result['data']['backtest']['total_trades']}")
        print(f"Win Rate: {result['data']['backtest']['win_rate']:.2f}%")
        print(f"Total Return: {result['data']['backtest']['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['data']['backtest']['sharpe_ratio']:.4f}")
    else:
        print(f"\n❌ ERROR: {response.json()}")
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to server. Is it running?")
    print("Run: python main.py")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
