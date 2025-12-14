import { motion } from 'framer-motion';
import { TrendingUp, Activity, BarChart3, Waves } from 'lucide-react';
import type { ExampleStrategy } from '@/types/strategy';
import { cn } from '@/lib/utils';

const exampleStrategies: ExampleStrategy[] = [
  {
    id: 'sma-crossover',
    name: 'SMA Crossover Strategy',
    description: 'Classic moving average crossover',
    text: 'Buy when the 10-period SMA crosses above the 50-period SMA. Sell when the 10-period SMA crosses below the 50-period SMA.',
    icon: 'trending',
  },
  {
    id: 'rsi-volume',
    name: 'RSI + Volume Momentum',
    description: 'RSI with volume confirmation',
    text: 'Buy when RSI is above 70 and volume is above average. Sell when RSI drops below 30 or volume decreases significantly.',
    icon: 'activity',
  },
  {
    id: 'bollinger-reversion',
    name: 'Bollinger Bands Mean Reversion',
    description: 'Mean reversion using Bollinger Bands',
    text: 'Buy when price touches the lower Bollinger Band and RSI is below 30. Sell when price reaches the middle band or upper band.',
    icon: 'waves',
  },
  {
    id: 'macd-divergence',
    name: 'MACD Divergence',
    description: 'MACD signal line crossover',
    text: 'Buy when MACD line crosses above the signal line and histogram is positive. Sell when MACD crosses below signal line.',
    icon: 'chart',
  },
];

const iconMap = {
  trending: TrendingUp,
  activity: Activity,
  chart: BarChart3,
  waves: Waves,
};

interface ExampleStrategiesProps {
  onSelect: (text: string) => void;
  disabled: boolean;
}

export function ExampleStrategies({ onSelect, disabled }: ExampleStrategiesProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="space-y-4"
    >
      <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
        Example Strategies
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {exampleStrategies.map((strategy, index) => {
          const Icon = iconMap[strategy.icon as keyof typeof iconMap];
          return (
            <motion.button
              key={strategy.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
              onClick={() => onSelect(strategy.text)}
              disabled={disabled}
              className={cn(
                "group p-4 rounded-xl text-left",
                "glass glass-hover",
                "transition-all duration-300",
                "hover:scale-[1.02] hover:glow-primary",
                "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              )}
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors duration-300">
                  <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-foreground truncate">
                    {strategy.name}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {strategy.description}
                  </p>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}
