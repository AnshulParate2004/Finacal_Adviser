import { motion } from 'framer-motion';
import { DollarSign, TrendingUp, TrendingDown, AlertTriangle, Zap, BarChart } from 'lucide-react';
import type { PerformanceMetrics as PerformanceMetricsType } from '@/types/strategy';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface PerformanceMetricsProps {
  metrics: PerformanceMetricsType;
}

export function PerformanceMetrics({ metrics }: PerformanceMetricsProps) {
  const metricItems = [
    {
      label: 'Total Profit',
      value: `$${metrics.total_profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      icon: DollarSign,
      color: metrics.total_profit >= 0 ? 'profit' : 'loss',
      tooltip: 'Net profit/loss from all trades',
    },
    {
      label: 'Winning Trades',
      value: metrics.winning_trades.toString(),
      icon: TrendingUp,
      color: 'profit',
      tooltip: 'Number of profitable trades',
    },
    {
      label: 'Losing Trades',
      value: metrics.losing_trades.toString(),
      icon: TrendingDown,
      color: 'loss',
      tooltip: 'Number of unprofitable trades',
    },
    {
      label: 'Max Drawdown',
      value: `${metrics.max_drawdown.toFixed(2)}%`,
      icon: AlertTriangle,
      color: 'warning',
      tooltip: 'Maximum peak-to-trough decline in portfolio value',
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpe_ratio.toFixed(2),
      icon: Zap,
      color: metrics.sharpe_ratio >= 1 ? 'profit' : metrics.sharpe_ratio >= 0 ? 'warning' : 'loss',
      tooltip: 'Risk-adjusted return. Above 1 is good, above 2 is excellent',
    },
    {
      label: 'Avg Trade Return',
      value: `${metrics.average_trade_return >= 0 ? '+' : ''}${metrics.average_trade_return.toFixed(2)}%`,
      icon: BarChart,
      color: metrics.average_trade_return >= 0 ? 'profit' : 'loss',
      tooltip: 'Average percentage return per trade',
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="glass rounded-xl overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-border/50">
        <h3 className="font-semibold text-foreground">Performance Metrics</h3>
      </div>
      <div className="divide-y divide-border/30">
        {metricItems.map((item, index) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + index * 0.05 }}
            className="flex items-center justify-between px-6 py-4 hover:bg-secondary/30 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div
                className={cn(
                  "p-2 rounded-lg",
                  item.color === 'profit' && "bg-profit/10 text-profit",
                  item.color === 'loss' && "bg-loss/10 text-loss",
                  item.color === 'warning' && "bg-warning/10 text-warning"
                )}
              >
                <item.icon className="w-4 h-4" />
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-sm text-muted-foreground cursor-help">
                    {item.label}
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{item.tooltip}</p>
                </TooltipContent>
              </Tooltip>
            </div>
            <span
              className={cn(
                "font-mono font-semibold",
                item.color === 'profit' && "text-profit",
                item.color === 'loss' && "text-loss",
                item.color === 'warning' && "text-warning"
              )}
            >
              {item.value}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
