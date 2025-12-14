import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import type { BacktestResult } from '@/types/strategy';
import { SummaryCards } from './SummaryCards';
import { PerformanceMetrics } from './PerformanceMetrics';
import { StrategyDetails } from './StrategyDetails';
import { TradeHistory } from './TradeHistory';
import { DataPeriodInfo } from './DataPeriodInfo';
import { Button } from '@/components/ui/button';

interface ResultsDisplayProps {
  result: BacktestResult;
  onReset: () => void;
}

export function ResultsDisplay({ result, onReset }: ResultsDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header with Reset */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          onClick={onReset}
          className="text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Analyze Another Strategy
        </Button>
      </div>

      {/* Summary Cards */}
      <SummaryCards
        totalReturn={result.total_return}
        winRate={result.win_rate}
        totalTrades={result.total_trades}
      />

      {/* Metrics and Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PerformanceMetrics metrics={result.performance} />
        <StrategyDetails details={result.strategy_details} />
      </div>

      {/* Trade History */}
      <TradeHistory trades={result.trades} />

      {/* Data Period Footer */}
      <DataPeriodInfo
        startDate={result.data_start_date}
        endDate={result.data_end_date}
        dataBars={result.data_bars}
      />
    </motion.div>
  );
}
