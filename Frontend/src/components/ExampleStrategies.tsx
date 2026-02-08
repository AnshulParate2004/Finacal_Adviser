import { motion } from 'framer-motion';
import { Lightbulb, ArrowRight, TrendingUp, Shield, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface ExampleStrategiesProps {
  onSelect: (text: string) => void;
  disabled: boolean;
}

const strategies = [
  {
    icon: TrendingUp,
    title: "Momentum RSI",
    description: "Buy when RSI > 70 and volume is high. Sell when RSI < 30.",
    color: "from-blue-500 to-cyan-500"
  },
  {
    icon: Shield,
    title: "Safe Moving Average",
    description: "Buy when price crosses above SMA 200. risk 1% per trade.",
    color: "from-emerald-500 to-green-500"
  },
  {
    icon: Activity,
    title: "Volatility Breakout",
    description: "Buy if price > Bollinger Upper Band. Stop loss at 2% below entry.",
    color: "from-purple-500 to-pink-500"
  }
];

export function ExampleStrategies({ onSelect, disabled }: ExampleStrategiesProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-muted-foreground px-1">
        <Lightbulb className="w-5 h-5 text-yellow-500" />
        <span className="text-sm font-medium">Try these examples</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {strategies.map((strategy, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 + 0.5 }}
          >
            <Button
              variant="outline"
              className={cn(
                "w-full h-auto p-6 flex flex-col items-start gap-4 whitespace-normal text-left relative overflow-hidden group",
                "bg-card/40 backdrop-blur-sm border-white/5 hover:border-white/10",
                "hover:bg-card/60 transition-all duration-500 hover:-translate-y-1"
              )}
              onClick={() => onSelect(strategy.description)}
              disabled={disabled}
            >
              <div className={cn(
                "absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-500 bg-gradient-to-br",
                strategy.color,
                "opacity-[0.03]"
              )} />

              <div className="flex items-center justify-between w-full relative z-10">
                <div className={cn(
                  "p-2.5 rounded-xl bg-gradient-to-br opacity-80 group-hover:opacity-100 transition-opacity",
                  strategy.color,
                  "text-white shadow-lg"
                )}>
                  <strategy.icon className="w-5 h-5" />
                </div>
                <ArrowRight className="w-4 h-4 text-muted-foreground/50 group-hover:translate-x-1 transition-transform" />
              </div>

              <div className="space-y-2 relative z-10">
                <div className="font-semibold text-foreground group-hover:text-primary transition-colors">
                  {strategy.title}
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2">
                  "{strategy.description}"
                </p>
              </div>
            </Button>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
