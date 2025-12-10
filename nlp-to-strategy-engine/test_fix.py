"""
Quick test to verify the data type fix
"""
import pandas as pd
import numpy as np
import talib

# Load the sample data
data = pd.read_csv('data/sample_data.csv', index_col='date', parse_dates=True)

print("="*80)
print("BEFORE FIX - Data Types:")
print("="*80)
print(data.dtypes)
print("\nFirst few rows:")
print(data.head())

# Try to use TA-Lib without conversion (should fail)
print("\n" + "="*80)
print("Testing TA-Lib WITHOUT conversion:")
print("="*80)
try:
    result = talib.SMA(data['close'].values, timeperiod=10)
    print("✓ SUCCESS - TA-Lib worked without conversion")
except Exception as e:
    print(f"✗ FAILED - Error: {e}")

print("\n" + "="*80)
print("AFTER FIX - Converting to float64:")
print("="*80)

# Convert to float64
numeric_columns = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].astype('float64')

print(data.dtypes)

# Try to use TA-Lib after conversion (should succeed)
print("\n" + "="*80)
print("Testing TA-Lib WITH conversion:")
print("="*80)
try:
    result = talib.SMA(data['close'].values, timeperiod=10)
    print("✓ SUCCESS - TA-Lib worked after conversion")
    print(f"  SMA calculated for {len(result)} bars")
    print(f"  First 5 SMA values: {result[:5]}")
    print(f"  Last 5 SMA values: {result[-5:]}")
except Exception as e:
    print(f"✗ FAILED - Error: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
