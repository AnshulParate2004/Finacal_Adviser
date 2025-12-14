import { motion } from 'framer-motion';
import { LineChart, BarChart3, TrendingUp, Activity } from 'lucide-react';

const loadingSteps = [
  { icon: LineChart, text: 'Parsing strategy rules...' },
  { icon: BarChart3, text: 'Loading historical data...' },
  { icon: TrendingUp, text: 'Running backtest simulation...' },
  { icon: Activity, text: 'Calculating performance metrics...' },
];

export function LoadingState() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="glass rounded-2xl p-12 text-center"
    >
      {/* Animated Logo */}
      <div className="relative w-24 h-24 mx-auto mb-8">
        <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
        <div className="absolute inset-2 rounded-full bg-primary/30 animate-pulse" />
        <div className="absolute inset-4 rounded-full bg-primary/40 flex items-center justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <TrendingUp className="w-8 h-8 text-primary" />
          </motion.div>
        </div>
      </div>

      <h2 className="text-2xl font-bold text-foreground mb-2">
        Analyzing Your Strategy
      </h2>
      <p className="text-muted-foreground mb-8">
        Please wait while we process your trading rules...
      </p>

      {/* Loading Steps */}
      <div className="space-y-3 max-w-sm mx-auto">
        {loadingSteps.map((step, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.5, duration: 0.3 }}
            className="flex items-center gap-3 text-left"
          >
            <div className="p-2 rounded-lg bg-secondary">
              <step.icon className="w-4 h-4 text-primary" />
            </div>
            <span className="text-sm text-muted-foreground">{step.text}</span>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 0] }}
              transition={{ delay: index * 0.5, duration: 1.5, repeat: Infinity }}
              className="ml-auto text-primary text-sm"
            >
              ●
            </motion.div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
