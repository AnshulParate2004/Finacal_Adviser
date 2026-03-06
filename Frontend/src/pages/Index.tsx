import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { BacktestResult } from '@/types/strategy';
import { StrategyInput } from '@/components/StrategyInput';
import { ExampleStrategies } from '@/components/ExampleStrategies';
import { LoadingState } from '@/components/LoadingState';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import { useToast } from '@/hooks/use-toast';

const API_URL = 'http://localhost:8000/api/strategy';
const CHECK_URL = 'http://localhost:8000/api/check-completeness';

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
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const totalScroll = document.documentElement.scrollTop;
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scroll = totalScroll / windowHeight;
      setScrollProgress(scroll);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const checkCompleteness = async (text: string): Promise<boolean> => {
    const formData = new FormData();
    formData.append('text', text);

    try {
      const response = await fetch(CHECK_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) return true;

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
      return true;
    }
  };

  const runStrategyAnalysis = async (text: string, symbol: { symbol: string; type: string } | null) => {
    const formData = new FormData();
    formData.append('text', text);
    if (symbol) {
      formData.append('symbol', symbol.symbol);
      formData.append('symbol_type', symbol.type);
    }

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

  const handleSubmit = async (text: string, symbol: { symbol: string; type: string } | null) => {
    setViewState('loading');
    setError('');
    setCompletenessError(null);

    try {
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

      const apiResponse = await runStrategyAnalysis(text, symbol);

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
          dsl_code: input.dsl_code,
          ast_tree: input.ast_tree,
        },
        trades: backtest.trades?.map((t: any) => ({
          entry_date: t.entry_date,
          entry_price: t.entry_price,
          exit_date: t.exit_date,
          exit_price: t.exit_price,
          profit_loss: t.profit_loss,
          profit_loss_percent: t.profit_loss_pct || t.profit_loss_percent,
          type: t.type || 'long',
        })) || [],
        data_start_date: backtest.data_start || '',
        data_end_date: backtest.data_end || '',
        data_bars: backtest.data_bars || 0,
        symbol: input.symbol || null,
        symbol_type: input.symbol_type || null,
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
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden relative selection:bg-primary/30">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-b from-primary/10 via-background to-background" />
        <div className="absolute -top-[20%] -right-[10%] w-[50%] h-[50%] rounded-full bg-accent/5 blur-3xl animate-float" />
        <div className="absolute top-[20%] -left-[10%] w-[40%] h-[40%] rounded-full bg-primary/5 blur-3xl animate-float" style={{ animationDelay: '2s' }} />

        {/* Grid Pattern */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(hsl(var(--foreground)) 1px, transparent 1px),
                             linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px)`,
            backgroundSize: '40px 40px',
            maskImage: 'linear-gradient(to bottom, black, transparent)',
            WebkitMaskImage: 'linear-gradient(to bottom, black, transparent)',
          }}
        />
      </div>

      <main className="relative z-10 container max-w-7xl mx-auto px-4 py-8 md:py-16">
        <AnimatePresence mode="wait">
          {viewState === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center justify-center min-h-[60vh]"
            >
              <LoadingState />
            </motion.div>
          )}

          {viewState === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20, filter: 'blur(10px)' }}
              transition={{ duration: 0.5 }}
              className="space-y-12 max-w-4xl mx-auto"
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
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex items-center justify-center min-h-[50vh]"
            >
              <ErrorDisplay message={error} onRetry={handleRetry} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Scroll Progress Bar */}
      <motion.div
        className="fixed bottom-0 left-0 h-1 bg-gradient-to-r from-primary to-accent z-50 origin-left"
        style={{ scaleX: scrollProgress }}
      />
    </div>
  );
};

export default Index;
