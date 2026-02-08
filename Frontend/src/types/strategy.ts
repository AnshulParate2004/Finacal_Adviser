export interface Trade {
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  profit_loss: number;
  profit_loss_percent: number;
  type: 'long' | 'short';
}

export interface StrategyDetails {
  original_text: string;
  indicators: string[];
  entry_conditions: string[];
  exit_conditions: string[];
  complexity: 'simple' | 'medium' | 'complex';
}

export interface PerformanceMetrics {
  total_profit: number;
  winning_trades: number;
  losing_trades: number;
  max_drawdown: number;
  sharpe_ratio: number;
  average_trade_return: number;
}

export interface BacktestResult {
  total_return: number;
  win_rate: number;
  total_trades: number;
  performance: PerformanceMetrics;
  strategy_details: StrategyDetails;
  trades: Trade[];
  data_start_date: string;
  data_end_date: string;
  data_bars: number;
}

export interface ExampleStrategy {
  id: string;
  name: string;
  description: string;
  text: string;
  icon: string;
}

export interface StrategyRequest {
  text: string;
  initial_capital: number;
  position_size: number;
}
