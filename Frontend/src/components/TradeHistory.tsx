import { motion } from 'framer-motion';
import { History, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import type { Trade } from '@/types/strategy';
import { cn } from '@/lib/utils';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

interface TradeHistoryProps {
  trades: Trade[];
}

export function TradeHistory({ trades }: TradeHistoryProps) {
  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="glass rounded-xl overflow-hidden"
    >
      <div className="px-6 py-4 border-b border-border/50 flex items-center gap-2">
        <History className="w-5 h-5 text-primary" />
        <h3 className="font-semibold text-foreground">Trade History</h3>
        <span className="ml-auto text-sm text-muted-foreground">
          {trades.length} trades
        </span>
      </div>

      <div className="max-h-96 overflow-y-auto scrollbar-thin">
        <Table>
          <TableHeader>
            <TableRow className="border-border/30 hover:bg-transparent">
              <TableHead className="text-muted-foreground font-medium">Entry Date</TableHead>
              <TableHead className="text-muted-foreground font-medium text-right">Entry Price</TableHead>
              <TableHead className="text-muted-foreground font-medium">Exit Date</TableHead>
              <TableHead className="text-muted-foreground font-medium text-right">Exit Price</TableHead>
              <TableHead className="text-muted-foreground font-medium text-right">Profit/Loss</TableHead>
              <TableHead className="text-muted-foreground font-medium text-right">Return %</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trades.map((trade, index) => {
              const isProfitable = trade.profit_loss >= 0;
              return (
                <motion.tr
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.02 }}
                  className={cn(
                    "border-border/20 transition-colors",
                    isProfitable ? "hover:bg-profit/5" : "hover:bg-loss/5"
                  )}
                >
                  <TableCell className="font-mono text-sm">
                    {formatDate(trade.entry_date)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {formatCurrency(trade.entry_price)}
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {formatDate(trade.exit_date)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {formatCurrency(trade.exit_price)}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className={cn(
                      "flex items-center justify-end gap-1 font-mono text-sm font-semibold",
                      isProfitable ? "text-profit" : "text-loss"
                    )}>
                      {isProfitable ? (
                        <ArrowUpRight className="w-4 h-4" />
                      ) : (
                        <ArrowDownRight className="w-4 h-4" />
                      )}
                      {formatCurrency(Math.abs(trade.profit_loss))}
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className={cn(
                      "font-mono text-sm font-semibold px-2 py-1 rounded",
                      isProfitable ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"
                    )}>
                      {isProfitable ? '+' : ''}{trade.profit_loss_percent.toFixed(2)}%
                    </span>
                  </TableCell>
                </motion.tr>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </motion.div>
  );
}
