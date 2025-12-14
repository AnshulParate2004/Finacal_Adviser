import { motion } from 'framer-motion';
import { FileText, Tag, ArrowRightCircle, ArrowLeftCircle, Gauge } from 'lucide-react';
import type { StrategyDetails as StrategyDetailsType } from '@/types/strategy';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';

interface StrategyDetailsProps {
  details: StrategyDetailsType;
}

export function StrategyDetails({ details }: StrategyDetailsProps) {
  const complexityColor = {
    simple: 'bg-profit/10 text-profit border-profit/20',
    medium: 'bg-warning/10 text-warning border-warning/20',
    complex: 'bg-primary/10 text-primary border-primary/20',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass rounded-xl overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-border/50 flex items-center justify-between">
        <h3 className="font-semibold text-foreground">Strategy Details</h3>
        <Badge className={cn("capitalize", complexityColor[details.complexity])}>
          <Gauge className="w-3 h-3 mr-1" />
          {details.complexity}
        </Badge>
      </div>

      <div className="p-6 space-y-6">
        {/* Original Text */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <FileText className="w-4 h-4" />
            Original Strategy
          </div>
          <p className="text-sm text-foreground bg-secondary/50 rounded-lg p-4 italic">
            "{details.original_text}"
          </p>
        </div>

        {/* Indicators */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Tag className="w-4 h-4" />
            Detected Indicators
          </div>
          <div className="flex flex-wrap gap-2">
            {details.indicators.map((indicator, index) => (
              <motion.div
                key={indicator}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 + index * 0.05 }}
              >
                <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20">
                  {indicator}
                </Badge>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Entry/Exit Conditions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Entry Conditions */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-profit">
              <ArrowRightCircle className="w-4 h-4" />
              Entry Conditions
            </div>
            <ul className="space-y-2">
              {details.entry_conditions.map((condition, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.05 }}
                  className="flex items-start gap-2 text-sm text-muted-foreground"
                >
                  <span className="text-profit mt-1">•</span>
                  <span>{condition}</span>
                </motion.li>
              ))}
            </ul>
          </div>

          {/* Exit Conditions */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-loss">
              <ArrowLeftCircle className="w-4 h-4" />
              Exit Conditions
            </div>
            <ul className="space-y-2">
              {details.exit_conditions.map((condition, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.05 }}
                  className="flex items-start gap-2 text-sm text-muted-foreground"
                >
                  <span className="text-loss mt-1">•</span>
                  <span>{condition}</span>
                </motion.li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
