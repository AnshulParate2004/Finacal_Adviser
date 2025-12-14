import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { BacktestResult, StrategyRequest } from '@/types/strategy';
import { StrategyInput } from '@/components/StrategyInput';
import { ExampleStrategies } from '@/components/ExampleStrategies';
import { LoadingState } from '@/components/LoadingState';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { useToast } from '@/hooks/use-toast';

const API_URL = 'http://localhost:8000/api/strategy';
const CHECK_URL = 'http://localhost:8000/api/check-completeness';

// Mock data for demo when API is unavailable
const mockResult: BacktestResult = {
  total_return: 24.67,
  win_rate: 62.5,
  total_trades: 48,
  performance: {
    total_profit: 2467.32,
    winning_trades: 30,
    losing_trades: 18,
    max_drawdown: 12.34,
    sharpe_ratio: 1.85,
    average_trade_return: 0.51,
  },
  strategy_details: {
    original_text: 'Buy when RSI is above 70 and volume is high. Sell when RSI drops below 30.',
    indicators: ['RSI', 'Volume'],
    entry_conditions: ['RSI crosses above 70', 'Volume exceeds 20-period average'],
    exit_conditions: ['RSI drops below 30', 'Stop loss at -5%'],
    complexity: 'medium',
  },
  trades: [
    { entry_date: '2024-01-15', entry_price: 152.34, exit_date: '2024-01-22', exit_price: 158.67, profit_loss: 633.0, profit_loss_percent: 4.16, type: 'long' },
    { entry_date: '2024-02-03', entry_price: 161.20, exit_date: '2024-02-08', exit_price: 157.45, profit_loss: -375.0, profit_loss_percent: -2.33, type: 'long' },
    { entry_date: '2024-02-15', entry_price: 155.80, exit_date: '2024-02-28', exit_price: 168.92, profit_loss: 1312.0, profit_loss_percent: 8.42, type: 'long' },
    { entry_date: '2024-03-05', entry_price: 172.45, exit_date: '2024-03-12', exit_price: 169.23, profit_loss: -322.0, profit_loss_percent: -1.87, type: 'long' },
    { entry_date: '2024-03-20', entry_price: 165.30, exit_date: '2024-04-02', exit_price: 178.56, profit_loss: 1326.0, profit_loss_percent: 8.02, type: 'long' },
    { entry_date: '2024-04-10', entry_price: 182.15, exit_date: '2024-04-15', exit_price: 176.89, profit_loss: -526.0, profit_loss_percent: -2.89, type: 'long' },
    { entry_date: '2024-04-22', entry_price: 174.50, exit_date: '2024-05-03', exit_price: 185.23, profit_loss: 1073.0, profit_loss_percent: 6.15, type: 'long' },
    { entry_date: '2024-05-10', entry_price: 188.90, exit_date: '2024-05-18', exit_price: 192.34, profit_loss: 344.0, profit_loss_percent: 1.82, type: 'long' },
  ],
  data_start_date: '2023-01-01',
  data_end_date: '2024-12-01',
  data_bars: 5234,
};

type ViewState = 'input' | 'loading' | 'results' | 'error';

interface CompletenessError {
  missing_elements: string[];
  suggestion: string;
}

const Index = () => {
  const [viewState, setViewState] = useState<ViewState>('input');
  const [strategyText, setStrategyText] = useState('');
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string>('');
  const [completenessError, setCompletenessError] = useState<CompletenessError | null>(null);
  const { toast } = useToast();

  const checkCompleteness = async (text: string): Promise<boolean> => {
    const formData = new FormData();
    formData.append('text', text);

    try {
      const response = await fetch(CHECK_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // If check endpoint fails, proceed anyway
        return true;
      }

      const data = await response.json();
      
      if (!data.is_complete) {
        setCompletenessError({
          missing_elements: data.missing_elements || [],
          suggestion: data.suggestion || 'Please add more details to your strategy.',
        });
        return false;
      }

      return true;
    } catch {
      // If check fails, proceed with strategy anyway
      return true;
    }
  };

  const runStrategyAnalysis = async (text: string) => {
    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    const apiResponse = await response.json();
    
    if (!apiResponse.success) {
      throw new Error(apiResponse.error || 'Strategy processing failed');
    }
    
    return apiResponse;
  };

  const handleSubmit = async (text: string) => {
    setViewState('loading');
    setError('');
    setCompletenessError(null);

    try {
      // Step 1: Check completeness first
      const isComplete = await checkCompleteness(text);
      
      if (!isComplete) {
        setViewState('input');
        toast({
          title: 'Incomplete Strategy',
          description: 'Please add more details to your strategy.',
          variant: 'destructive',
        });
        return;
      }

      // Step 2: If complete, run strategy analysis automatically
      const apiResponse = await runStrategyAnalysis(text);
      
      // Transform API response to our BacktestResult format
      const backtest = apiResponse.data.backtest;
      const input = apiResponse.data.input;
      
      const transformedResult: BacktestResult = {
        total_return: backtest.total_return_pct,
        win_rate: backtest.win_rate,
        total_trades: backtest.total_trades,
        performance: {
          total_profit: backtest.total_profit,
          winning_trades: backtest.winning_trades,
          losing_trades: backtest.losing_trades,
          max_drawdown: Math.abs(backtest.max_drawdown),
          sharpe_ratio: backtest.sharpe_ratio,
          average_trade_return: backtest.avg_trade_return,
        },
        strategy_details: {
          original_text: input.original_text,
          indicators: input.indicators_used || [],
          entry_conditions: input.parsed_rule.entry?.map((c: any) => 
            `${c.left} ${c.operator} ${c.right}`
          ) || [],
          exit_conditions: input.parsed_rule.exit?.map((c: any) => 
            `${c.left} ${c.operator} ${c.right}`
          ) || [],
          complexity: input.complexity as 'simple' | 'medium' | 'complex',
        },
        trades: backtest.trades?.map((t: any) => ({
          entry_date: t.entry_date,
          entry_price: t.entry_price,
          exit_date: t.exit_date,
          exit_price: t.exit_price,
          profit_loss: t.profit_loss,
          profit_loss_percent: t.profit_loss_pct || t.profit_loss_percent,
          type: t.type || 'long' as const,
        })) || [],
        data_start_date: backtest.data_start || '',
        data_end_date: backtest.data_end || '',
        data_bars: backtest.data_bars || 0,
      };
      
      setResult(transformedResult);
      setViewState('results');
      setCompletenessError(null);
      toast({
        title: 'Analysis Complete',
        description: `Analyzed ${backtest.total_trades} trades with ${backtest.win_rate.toFixed(1)}% win rate.`,
      });
    } catch (err) {
      console.error('API Error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to connect to backend';
      setError(
        errorMessage.includes('Failed to fetch') 
          ? 'Cannot connect to backend. Make sure your server is running at http://localhost:8000'
          : errorMessage
      );
      setViewState('error');
      toast({
        title: 'Connection Failed',
        description: 'Could not reach the backend server.',
        variant: 'destructive',
      });
    }
  };

  const handleReset = () => {
    setViewState('input');
    setResult(null);
    setError('');
    setCompletenessError(null);
  };

  const handleRetry = () => {
    setViewState('input');
    setError('');
    setCompletenessError(null);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-primary/5 via-background to-background pointer-events-none" />
      
      {/* Grid pattern overlay */}
      <div 
        className="fixed inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(hsl(var(--foreground)) 1px, transparent 1px),
                           linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px)`,
          backgroundSize: '50px 50px',
        }}
      />

      <main className="relative z-10 container max-w-6xl mx-auto px-4 py-8 md:py-12">
        <AnimatePresence mode="wait">
          {viewState === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <LoadingState />
            </motion.div>
          )}

          {viewState === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-8"
            >
              <StrategyInput
                onSubmit={handleSubmit}
                isLoading={false}
                strategyText={strategyText}
                onTextChange={setStrategyText}
                completenessError={completenessError}
              />
              <ExampleStrategies
                onSelect={setStrategyText}
                disabled={false}
              />
            </motion.div>
          )}

          {viewState === 'results' && result && (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ResultsDisplay result={result} onReset={handleReset} />
            </motion.div>
          )}

          {viewState === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ErrorDisplay message={error} onRetry={handleRetry} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default Index;
