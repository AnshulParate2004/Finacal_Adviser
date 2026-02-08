import { motion } from 'framer-motion';
import { TrendingUp, Target, BarChart2, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface SummaryCardsProps {
  totalReturn: number;
  winRate: number;
  totalTrades: number;
}

export function SummaryCards({ totalReturn, winRate, totalTrades }: SummaryCardsProps) {
  const cards = [
    {
      label: "Total Return",
      value: `${totalReturn > 0 ? '+' : ''}${totalReturn.toFixed(2)}%`,
      icon: TrendingUp,
      color: totalReturn >= 0 ? "text-success" : "text-destructive",
      bgClass: totalReturn >= 0 ? "from-success/20 to-success/5 border-success/20" : "from-destructive/20 to-destructive/5 border-destructive/20",
      trend: true
    },
    {
      label: "Win Rate",
      value: `${winRate.toFixed(1)}%`,
      icon: Target,
      color: "text-primary",
      bgClass: "from-primary/20 to-primary/5 border-primary/20",
      trend: false
    },
    {
      label: "Total Trades",
      value: totalTrades.toString(),
      icon: BarChart2,
      color: "text-accent-foreground",
      bgClass: "from-indigo-500/20 to-indigo-500/5 border-indigo-500/20",
      trend: false
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {cards.map((card, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          whileHover={{ y: -5, transition: { duration: 0.2 } }}
          className="relative group"
        >
          <div className={`absolute inset-0 bg-gradient-to-br ${card.bgClass} opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl blur-xl`} />
          <Card className="relative overflow-hidden border-white/5 bg-card/40 backdrop-blur-xl p-6 h-full shadow-lg group-hover:border-white/10 transition-colors">
            <div className={`absolute top-0 right-0 p-24 bg-gradient-to-br ${card.bgClass} opacity-[0.05] rounded-full blur-2xl -translate-y-10 translate-x-10`} />

            <div className="flex items-center justify-between mb-4 relative z-10">
              <div className="flex items-center gap-2">
                <div className={`p-2.5 rounded-xl bg-white/5 ${card.color} shadow-inner`}>
                  <card.icon className="w-5 h-5" />
                </div>
                <span className="text-sm font-medium text-muted-foreground">{card.label}</span>
              </div>
              {card.trend && (
                <div className={`flex items-center text-xs font-bold px-2 py-1 rounded-full bg-white/5 ${card.color}`}>
                  {totalReturn >= 0 ? <ArrowUpRight className="w-3 h-3 mr-1" /> : <ArrowDownRight className="w-3 h-3 mr-1" />}
                  {totalReturn >= 0 ? 'PROFIT' : 'LOSS'}
                </div>
              )}
            </div>

            <div className="relative z-10">
              <div className={`text-4xl font-bold tracking-tight ${card.color} text-glow`}>
                {card.value}
              </div>
            </div>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}
