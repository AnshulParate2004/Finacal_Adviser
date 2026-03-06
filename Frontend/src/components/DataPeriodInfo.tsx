import { motion } from 'framer-motion';
import { Calendar, Database, BarChart3 } from 'lucide-react';

interface DataPeriodInfoProps {
  startDate: string;
  endDate: string;
  dataBars: number;
  symbol?: string | null;
  symbolType?: 'stock' | 'etf' | null;
}

export function DataPeriodInfo({ startDate, endDate, dataBars, symbol, symbolType }: DataPeriodInfoProps) {
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.6 }}
      className="flex flex-wrap items-center justify-center gap-6 py-4 text-sm text-muted-foreground"
    >
      {symbol && (
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-primary" />
          <span className="font-medium">{symbol}</span>
          <span className="text-xs capitalize">({symbolType || 'stock'})</span>
        </div>
      )}
      <div className="flex items-center gap-2">
        <Calendar className="w-4 h-4 text-primary" />
        <span>
          {formatDate(startDate)} — {formatDate(endDate)}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <Database className="w-4 h-4 text-primary" />
        <span>{dataBars.toLocaleString()} data bars analyzed</span>
      </div>
    </motion.div>
  );
}
