import { motion } from 'framer-motion';
import { PerformanceMetrics as MetricsType } from '@/types/strategy';
import { Card } from '@/components/ui/card';
import { DollarSign, Percent, TrendingDown, TrendingUp, Activity, BarChart } from 'lucide-react';

interface PerformanceMetricsProps {
  metrics: MetricsType;
}

export function PerformanceMetrics({ metrics }: PerformanceMetricsProps) {
  const items = [
    { label: 'Total Profit', value: `$${metrics.total_profit.toFixed(2)}`, icon: DollarSign, color: metrics.total_profit >= 0 ? "text-success" : "text-destructive" },
    { label: 'Winning Trades', value: metrics.winning_trades, icon: TrendingUp, color: "text-success" },
    { label: 'Losing Trades', value: metrics.losing_trades, icon: TrendingDown, color: "text-destructive" },
    { label: 'Max Drawdown', value: `${metrics.max_drawdown.toFixed(2)}%`, icon: Activity, color: "text-warning" },
    { label: 'Sharpe Ratio', value: metrics.sharpe_ratio.toFixed(2), icon: BarChart, color: "text-primary" },
    { label: 'Avg Trade', value: `${metrics.average_trade_return > 0 ? '+' : ''}${metrics.average_trade_return.toFixed(2)}%`, icon: Percent, color: metrics.average_trade_return >= 0 ? "text-success" : "text-destructive" },
  ];

  return (
    <Card className="p-6 h-full glass-card overflow-hidden relative">
      <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-32 translate-x-32" />

      <h3 className="text-lg font-semibold mb-6 flex items-center gap-2 relative z-10">
        <Activity className="w-5 h-5 text-primary" />
        Key Performance Metrics
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-6 relative z-10">
        {items.map((item, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 + 0.3 }}
            className="space-y-2 p-3 rounded-xl hover:bg-white/5 transition-colors"
          >
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <item.icon className="w-4 h-4 opacity-70" />
              <span>{item.label}</span>
            </div>
            <div className={`text-xl font-bold tracking-tight ${item.color} filter drop-shadow-sm`}>
              {item.value}
            </div>
          </motion.div>
        ))}
      </div>
    </Card>
  );
}
