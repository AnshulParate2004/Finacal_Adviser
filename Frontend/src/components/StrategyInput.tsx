import { motion } from 'framer-motion';
import { Sparkles, Send, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

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
  const handleSubmit = () => {
    if (strategyText.trim()) {
      onSubmit(strategyText);
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
          Describe your trading strategy in plain English. Include capital and position size if needed (e.g., "with $50k capital investing 50%").
        </p>
      </div>

      {/* Completeness Error */}
      {completenessError && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl bg-destructive/10 border border-destructive/30 space-y-3"
        >
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="w-5 h-5" />
            <span className="font-semibold">Incomplete Strategy</span>
          </div>
          {completenessError.missing_elements.length > 0 && (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Missing elements:</p>
              <ul className="list-disc list-inside text-sm text-destructive/80 space-y-1">
                {completenessError.missing_elements.map((element, i) => (
                  <li key={i}>{element}</li>
                ))}
              </ul>
            </div>
          )}
          {completenessError.suggestion && (
            <p className="text-sm text-muted-foreground italic">
              💡 {completenessError.suggestion}
            </p>
          )}
        </motion.div>
      )}

      {/* Strategy Input */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-muted-foreground">Strategy Description</label>
        <textarea
          value={strategyText}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder="Buy when RSI is above 70 and volume is high. Sell when RSI drops below 30. Start with $50k capital investing 50%..."
          className={cn(
            "w-full h-32 px-4 py-3 rounded-xl",
            "bg-secondary/50 border border-border/50",
            "text-foreground placeholder:text-muted-foreground/50",
            "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50",
            "resize-none transition-all duration-200",
            "font-medium",
            completenessError && "border-destructive/50 focus:ring-destructive/50 focus:border-destructive/50"
          )}
          disabled={isLoading}
        />
        <p className="text-xs text-muted-foreground/70">
          Defaults: $10,000 capital, 100% position size if not specified
        </p>
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
