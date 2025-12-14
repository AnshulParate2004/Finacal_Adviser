import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, DollarSign, Percent, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';

interface StrategyInputProps {
  onSubmit: (text: string, capital: number, positionSize: number) => void;
  isLoading: boolean;
  strategyText: string;
  onTextChange: (text: string) => void;
}

export function StrategyInput({ onSubmit, isLoading, strategyText, onTextChange }: StrategyInputProps) {
  const [capital, setCapital] = useState(10000);
  const [positionSize, setPositionSize] = useState(100);

  const handleSubmit = () => {
    if (strategyText.trim()) {
      onSubmit(strategyText, capital, positionSize);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass rounded-2xl p-8 space-y-6"
    >
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="p-2.5 rounded-xl bg-primary/10 glow-primary">
            <Sparkles className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-3xl font-bold gradient-text">Trading Strategy Analyzer</h1>
        </div>
        <p className="text-muted-foreground max-w-xl mx-auto">
          Describe your trading strategy in plain English. Our AI will parse it, backtest it against historical data, and show you detailed performance metrics.
        </p>
      </div>

      {/* Strategy Input */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-muted-foreground">Strategy Description</label>
        <textarea
          value={strategyText}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder="Buy when RSI is above 70 and volume is high. Sell when RSI drops below 30..."
          className={cn(
            "w-full h-28 px-4 py-3 rounded-xl",
            "bg-secondary/50 border border-border/50",
            "text-foreground placeholder:text-muted-foreground/50",
            "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50",
            "resize-none transition-all duration-200",
            "font-medium"
          )}
          disabled={isLoading}
        />
      </div>

      {/* Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Initial Capital */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <DollarSign className="w-4 h-4" />
              Initial Capital
            </label>
            <span className="text-sm font-mono font-bold text-primary">
              ${capital.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value) || 10000)}
              className={cn(
                "w-full px-3 py-2 rounded-lg",
                "bg-secondary/50 border border-border/50",
                "text-foreground font-mono",
                "focus:outline-none focus:ring-2 focus:ring-primary/50",
                "transition-all duration-200"
              )}
              min={1000}
              step={1000}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Position Size */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <Percent className="w-4 h-4" />
              Position Size
            </label>
            <span className="text-sm font-mono font-bold text-primary">
              {positionSize}%
            </span>
          </div>
          <Slider
            value={[positionSize]}
            onValueChange={(value) => setPositionSize(value[0])}
            min={10}
            max={100}
            step={5}
            disabled={isLoading}
            className="py-2"
          />
        </div>
      </div>

      {/* Submit Button */}
      <Button
        onClick={handleSubmit}
        disabled={isLoading || !strategyText.trim()}
        className={cn(
          "w-full h-14 text-lg font-semibold",
          "bg-primary hover:bg-primary/90",
          "transition-all duration-300",
          !isLoading && strategyText.trim() && "glow-primary hover:scale-[1.02]"
        )}
      >
        {isLoading ? (
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            <span>Analyzing Strategy...</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Send className="w-5 h-5" />
            <span>Analyze Strategy</span>
          </div>
        )}
      </Button>
    </motion.div>
  );
}
