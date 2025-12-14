import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Target, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SummaryCardsProps {
  totalReturn: number;
  winRate: number;
  totalTrades: number;
}

export function SummaryCards({ totalReturn, winRate, totalTrades }: SummaryCardsProps) {
  const isPositive = totalReturn >= 0;

  const cards = [
    {
      label: 'Total Return',
      value: `${isPositive ? '+' : ''}${totalReturn.toFixed(2)}%`,
      icon: isPositive ? TrendingUp : TrendingDown,
      color: isPositive ? 'profit' : 'loss',
      glow: isPositive ? 'glow-success' : 'glow-loss',
    },
    {
      label: 'Win Rate',
      value: `${winRate.toFixed(1)}%`,
      icon: Target,
      color: winRate >= 50 ? 'profit' : 'loss',
      progress: winRate,
    },
    {
      label: 'Total Trades',
      value: totalTrades.toString(),
      icon: BarChart3,
      color: 'primary',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {cards.map((card, index) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className={cn(
            "glass rounded-xl p-6 relative overflow-hidden",
            card.glow
          )}
        >
          {/* Background glow effect */}
          <div
            className={cn(
              "absolute -right-4 -top-4 w-24 h-24 rounded-full opacity-20 blur-2xl",
              card.color === 'profit' && "bg-profit",
              card.color === 'loss' && "bg-loss",
              card.color === 'primary' && "bg-primary"
            )}
          />

          <div className="relative z-10">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-muted-foreground">
                {card.label}
              </span>
              <div
                className={cn(
                  "p-2 rounded-lg",
                  card.color === 'profit' && "bg-profit/10 text-profit",
                  card.color === 'loss' && "bg-loss/10 text-loss",
                  card.color === 'primary' && "bg-primary/10 text-primary"
                )}
              >
                <card.icon className="w-4 h-4" />
              </div>
            </div>

            <div
              className={cn(
                "text-3xl font-bold font-mono",
                card.color === 'profit' && "text-profit",
                card.color === 'loss' && "text-loss",
                card.color === 'primary' && "text-primary"
              )}
            >
              {card.value}
            </div>

            {card.progress !== undefined && (
              <div className="mt-3 h-1.5 bg-secondary rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${card.progress}%` }}
                  transition={{ delay: 0.5, duration: 0.8, ease: 'easeOut' }}
                  className={cn(
                    "h-full rounded-full",
                    card.color === 'profit' ? "bg-profit" : "bg-loss"
                  )}
                />
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
