import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Send, AlertCircle, TrendingUp, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useState } from 'react';

interface CompletenessError {
  missing_elements: string[];
  suggestion: string;
}

interface StrategyInputProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
  strategyText: string;
  onTextChange: (text: string) => void;
  completenessError?: CompletenessError | null;
}

export function StrategyInput({ onSubmit, isLoading, strategyText, onTextChange, completenessError }: StrategyInputProps) {
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = () => {
    if (strategyText.trim()) {
      onSubmit(strategyText);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="relative z-10"
    >
      <div className="glass-card rounded-2xl p-8 space-y-8 relative overflow-hidden group">
        {/* Animated Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

        {/* Header */}
        <div className="text-center space-y-4 relative z-10">
          <motion.div
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="flex items-center justify-center gap-3 mb-2"
          >
            <div className="p-3 rounded-2xl bg-primary/10 border border-primary/20 glow-box">
              <Sparkles className="w-6 h-6 text-primary animate-pulse" />
            </div>
            <h1 className="text-4xl font-bold gradient-text tracking-tight">Strategy Analyzer</h1>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-muted-foreground/80 max-w-xl mx-auto text-lg leading-relaxed"
          >
            Transform your trading ideas into backtested reality.
            <br />
            <span className="text-sm opacity-70">Powered by advanced NLP & quantitative analysis</span>
          </motion.p>
        </div>

        {/* Completeness Error */}
        <AnimatePresence>
          {completenessError && (
            <motion.div
              initial={{ opacity: 0, height: 0, marginBottom: 0 }}
              animate={{ opacity: 1, height: 'auto', marginBottom: 20 }}
              exit={{ opacity: 0, height: 0, marginBottom: 0 }}
              className="overflow-hidden"
            >
              <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/20 space-y-3 relative overflow-hidden">
                <div className="absolute inset-0 bg-destructive/5 animate-pulse" />
                <div className="flex items-start gap-3 relative z-10">
                  <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
                  <div className="space-y-2 flex-1">
                    <span className="font-semibold text-destructive">Strategy Requirements Missing</span>

                    {completenessError.missing_elements.length > 0 && (
                      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-muted-foreground">
                        {completenessError.missing_elements.map((element, i) => (
                          <li key={i} className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-destructive/60" />
                            {element}
                          </li>
                        ))}
                      </ul>
                    )}

                    {completenessError.suggestion && (
                      <div className="flex items-start gap-2 text-sm text-primary/80 bg-primary/5 p-3 rounded-lg border border-primary/10 mt-2">
                        <Info className="w-4 h-4 mt-0.5 shrink-0" />
                        <p className="italic">"{completenessError.suggestion}"</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Strategy Input */}
        <div className="space-y-4 relative group/input">
          <div className="flex items-center justify-between px-1">
            <label className={cn(
              "text-sm font-medium transition-colors duration-300",
              isFocused ? "text-primary ml-1" : "text-muted-foreground"
            )}>
              Strategy Description
            </label>
            <div className="flex items-center gap-2 text-xs text-muted-foreground/50">
              <TrendingUp className="w-3 h-3" />
              <span>Natural Language Supported</span>
            </div>
          </div>

          <div className="relative">
            <div className={cn(
              "absolute -inset-0.5 rounded-xl bg-gradient-to-r from-primary via-purple-500 to-pink-500 opacity-0 transition-opacity duration-300 blur-md",
              isFocused && "opacity-50"
            )} />

            <textarea
              value={strategyText}
              onChange={(e) => onTextChange(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Example: Buy when RSI is above 70 and volume is high. Sell when RSI drops below 30. Start with $50k capital..."
              className={cn(
                "w-full h-40 px-6 py-5 rounded-xl relative z-10",
                "bg-secondary/30 backdrop-blur-md border border-white/5",
                "text-foreground placeholder:text-muted-foreground/30",
                "focus:outline-none focus:bg-secondary/50",
                "resize-none transition-all duration-300",
                "text-lg font-light leading-relaxed font-mono",
                completenessError && "border-destructive/30 focus:border-destructive/50"
              )}
              disabled={isLoading}
            />
          </div>

          <div className="flex justify-between items-center text-xs text-muted-foreground/50 px-2 group-focus-within/input:text-muted-foreground/80 transition-colors">
            <p>Defaults: $10,000 capital, 100% position size</p>
            <p>{strategyText.length} chars</p>
          </div>
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={isLoading || !strategyText.trim()}
          className={cn(
            "w-full h-16 text-lg font-semibold rounded-xl relative overflow-hidden group/btn",
            "bg-primary hover:bg-primary/90 transition-all duration-300",
            !isLoading && strategyText.trim() && "glow-box hover:scale-[1.01]"
          )}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-200%] group-hover/btn:translate-x-[200%] transition-transform duration-1000" />

          {isLoading ? (
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span className="animate-pulse">Analyzing Market Data...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Send className="w-5 h-5 group-hover/btn:translate-x-1 transition-transform" />
              <span>Backtest Strategy</span>
            </div>
          )}
        </Button>
      </div>
    </motion.div>
  );
}
