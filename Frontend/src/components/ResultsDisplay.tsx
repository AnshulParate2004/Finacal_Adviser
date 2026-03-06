import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import type { BacktestResult } from '@/types/strategy';
import { SummaryCards } from './SummaryCards';
import { PerformanceMetrics } from './PerformanceMetrics';
import { StrategyDetails } from './StrategyDetails';
import { DslAstView } from './DslAstView';
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
      className="space-y-8 pb-12"
    >
      {/* Header with Reset */}
      <div className="flex items-center justify-between sticky top-4 z-50">
        <Button
          variant="outline"
          onClick={onReset}
          className="bg-background/50 backdrop-blur-md border-white/10 hover:bg-background/80"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          New Analysis
        </Button>
      </div>

      {/* Summary Cards */}
      <SummaryCards
        totalReturn={result.total_return}
        winRate={result.win_rate}
        totalTrades={result.total_trades}
      />

      {/* DSL & AST visualization */}
      <DslAstView
        dslCode={result.strategy_details.dsl_code}
        astTree={result.strategy_details.ast_tree}
      />

      {/* Metrics and Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <PerformanceMetrics metrics={result.performance} />
        </div>
        <div className="lg:col-span-5">
          <StrategyDetails details={result.strategy_details} />
        </div>
      </div>

      {/* Trade History */}
      <div className="space-y-4">
        <h3 className="text-xl font-semibold pl-1">Trade History</h3>
        <TradeHistory trades={result.trades} />
      </div>

      {/* Data Period Footer */}
      <DataPeriodInfo
        startDate={result.data_start_date}
        endDate={result.data_end_date}
        dataBars={result.data_bars}
      />
    </motion.div>
  );
}
