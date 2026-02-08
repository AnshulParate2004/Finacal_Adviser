import { motion } from 'framer-motion';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorDisplayProps {
  message: string;
  onRetry: () => void;
}

export function ErrorDisplay({ message, onRetry }: ErrorDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass rounded-2xl p-8 text-center max-w-md mx-auto"
    >
      <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-loss/10 flex items-center justify-center">
        <AlertTriangle className="w-8 h-8 text-loss" />
      </div>

      <h3 className="text-xl font-semibold text-foreground mb-2">
        Strategy Analysis Failed
      </h3>
      <p className="text-muted-foreground mb-6">
        {message}
      </p>

      <Button onClick={onRetry} variant="outline" className="gap-2">
        <RefreshCw className="w-4 h-4" />
        Try Again
      </Button>
    </motion.div>
  );
}
