import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { Trade } from '@/types/strategy';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ArrowUpCircle, ArrowDownCircle } from 'lucide-react';

interface TradeHistoryProps {
  trades: Trade[];
}

export function TradeHistory({ trades }: TradeHistoryProps) {
  return (
    <Card className="glass-card overflow-hidden border-white/5">
      <div className="max-h-[500px] overflow-auto scrollbar-thin">
        <Table>
          <TableHeader className="sticky top-0 bg-secondary/95 backdrop-blur-md z-20">
            <TableRow className="border-b border-white/10 hover:bg-transparent">
              <TableHead className="w-[100px] text-primary font-bold">Type</TableHead>
              <TableHead className="font-semibold text-foreground/80">Entry Date</TableHead>
              <TableHead className="font-semibold text-foreground/80">Entry Price</TableHead>
              <TableHead className="font-semibold text-foreground/80">Exit Date</TableHead>
              <TableHead className="font-semibold text-foreground/80">Exit Price</TableHead>
              <TableHead className="text-right font-semibold text-foreground/80">P/L</TableHead>
              <TableHead className="text-right font-semibold text-foreground/80">Return</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trades.map((trade, index) => (
              <motion.tr
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 + 0.5 }}
                className="group border-b border-white/5 hover:bg-white/5 transition-colors"
              >
                <TableCell className="font-medium">
                  <Badge
                    variant="outline"
                    className={trade.type === 'long'
                      ? "bg-blue-500/10 text-blue-400 border-blue-500/20 group-hover:border-blue-500/50 transition-colors"
                      : "bg-purple-500/10 text-purple-400 border-purple-500/20 group-hover:border-purple-500/50 transition-colors"
                    }
                  >
                    {trade.type.toUpperCase()}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">{format(new Date(trade.entry_date), 'MMM dd, yyyy')}</TableCell>
                <TableCell className="font-mono">${trade.entry_price.toFixed(2)}</TableCell>
                <TableCell className="text-muted-foreground">{format(new Date(trade.exit_date), 'MMM dd, yyyy')}</TableCell>
                <TableCell className="font-mono">${trade.exit_price.toFixed(2)}</TableCell>
                <TableCell className={`text-right font-bold font-mono ${trade.profit_loss >= 0 ? 'text-success' : 'text-destructive'}`}>
                  {trade.profit_loss >= 0 ? '+' : ''}{trade.profit_loss.toFixed(2)}
                </TableCell>
                <TableCell className="text-right">
                  <div className={`flex items-center justify-end gap-1 font-bold ${trade.profit_loss_percent >= 0 ? 'text-success' : 'text-destructive'}`}>
                    {trade.profit_loss_percent >= 0
                      ? <ArrowUpCircle className="w-4 h-4 opacity-50" />
                      : <ArrowDownCircle className="w-4 h-4 opacity-50" />
                    }
                    {trade.profit_loss_percent.toFixed(2)}%
                  </div>
                </TableCell>
              </motion.tr>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
