import { motion } from 'framer-motion';
import { StrategyDetails as DetailsType } from '@/types/strategy';
import { Card } from '@/components/ui/card';
import { ScrollText, Gauge, ArrowRightFromLine, ArrowLeftFromLine, Layers } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface StrategyDetailsProps {
  details: DetailsType;
}

export function StrategyDetails({ details }: StrategyDetailsProps) {
  return (
    <Card className="p-6 h-full glass-card flex flex-col relative overflow-hidden">
      {/* Decorative gradient blob */}
      <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-accent/5 rounded-full blur-3xl" />

      <h3 className="text-lg font-semibold mb-6 flex items-center gap-2 relative z-10">
        <ScrollText className="w-5 h-5 text-accent" />
        Strategy Breakdown
      </h3>

      <div className="space-y-6 flex-1 relative z-10">
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-primary font-medium">
            <Gauge className="w-4 h-4" />
            <span>Technical Indicators</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {details.indicators.map((indicator, i) => (
              <motion.div
                key={i}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.1 * i }}
              >
                <Badge variant="secondary" className="bg-secondary/50 border-white/5 hover:bg-secondary/80 px-3 py-1">
                  {indicator}
                </Badge>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-success font-medium">
            <ArrowRightFromLine className="w-4 h-4" />
            <span>Entry Conditions</span>
          </div>
          <ul className="space-y-2">
            {details.entry_conditions.map((condition, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * i }}
                className="text-sm text-muted-foreground flex items-start gap-2 bg-white/5 p-2 rounded-lg"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-success mt-1.5 shrink-0" />
                {condition}
              </motion.li>
            ))}
          </ul>
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-destructive font-medium">
            <ArrowLeftFromLine className="w-4 h-4" />
            <span>Exit Conditions</span>
          </div>
          <ul className="space-y-2">
            {details.exit_conditions.map((condition, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * i }}
                className="text-sm text-muted-foreground flex items-start gap-2 bg-white/5 p-2 rounded-lg"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-destructive mt-1.5 shrink-0" />
                {condition}
              </motion.li>
            ))}
          </ul>
        </div>

        <div className="pt-4 border-t border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Layers className="w-4 h-4" />
            <span className="text-sm">Complexity Score</span>
          </div>
          <Badge variant="outline" className={`capitalize ${details.complexity === 'simple' ? 'border-success text-success bg-success/10' :
              details.complexity === 'medium' ? 'border-warning text-warning bg-warning/10' :
                'border-destructive text-destructive bg-destructive/10'
            }`}>
            {details.complexity}
          </Badge>
        </div>
      </div>
    </Card>
  );
}
