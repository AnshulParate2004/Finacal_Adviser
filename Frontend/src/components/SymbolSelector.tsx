import { useState, useEffect, useMemo, useRef } from 'react';
import { ChevronDown, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

const API_BASE = 'http://localhost:8000';

export interface SymbolItem {
  symbol: string;
  security_name: string;
  type: 'stock' | 'etf';
}

interface SymbolSelectorProps {
  value: SymbolItem | null;
  onChange: (symbol: SymbolItem | null) => void;
  disabled?: boolean;
  className?: string;
}

type FilterType = 'stock' | 'etf';

function formatOption(s: SymbolItem) {
  return `${s.security_name} (${s.symbol})`;
}

function fileHint(s: SymbolItem) {
  const dir = s.type === 'etf' ? 'etfs' : 'stocks';
  return `archive/${dir}/${s.symbol}.csv`;
}

export function SymbolSelector({ value, onChange, disabled, className }: SymbolSelectorProps) {
  const [symbols, setSymbols] = useState<SymbolItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<FilterType>('stock');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const type = filter;
    setLoading(true);
    fetch(`${API_BASE}/api/symbols${type ? `?type=${type}` : ''}`)
      .then((r) => r.json())
      .then((d) => setSymbols(d.symbols || []))
      .catch(() => setSymbols([]))
      .finally(() => setLoading(false));
  }, [filter]);

  // Keep input in sync with selection: show Security Name (SYMBOL)
  useEffect(() => {
    if (value) setInputValue(formatOption(value));
    else setInputValue('');
  }, [value]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return symbols.slice(0, 200);
    return symbols
      .filter((s) => {
        const sym = s.symbol.toLowerCase();
        const name = s.security_name.toLowerCase();
        const display = formatOption(s).toLowerCase();
        return sym.includes(q) || name.includes(q) || display.includes(q) || q.includes(sym);
      })
      .slice(0, 100);
  }, [symbols, search]);

  const showDropdown = open && !disabled;

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        if (value) setInputValue(formatOption(value));
        else setInputValue('');
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [value]);

  const handleSelect = (s: SymbolItem | null) => {
    onChange(s);
    setOpen(false);
    if (s) setInputValue(formatOption(s));
    else setSearch('');
  };

  return (
    <div ref={containerRef} className={cn('space-y-2', className)}>
      <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
        <BarChart3 className="w-4 h-4" />
        Security — type name or symbol (e.g. Agilent or A)
      </label>

      {/* Filter tabs: ETFs = only ETF column Y, Stocks = only ETF column N */}
      <div className="flex gap-1 p-1 rounded-lg bg-white/5 border border-white/5 relative z-10">
        {(['stock', 'etf'] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setFilter(f);
            }}
            disabled={disabled}
            className={cn(
              'flex-1 py-1.5 px-2 rounded text-xs font-medium capitalize transition-colors',
              filter === f
                ? 'bg-primary/20 text-primary'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            {f === 'stock' ? 'Stocks' : 'ETFs'}
          </button>
        ))}
      </div>
      {filter === 'etf' && (
        <p className="text-xs text-muted-foreground">Showing only securities with ETF = Y</p>
      )}
      {filter === 'stock' && (
        <p className="text-xs text-muted-foreground">Showing only securities with ETF = N</p>
      )}

      {/* Type-to-search input + recommendations */}
      <div className="relative">
        <div className="relative flex items-center">
          <input
            ref={inputRef}
            type="text"
            placeholder="Type security name or symbol… (e.g. Agilent Technologies or A)"
            value={open ? search : inputValue}
            onChange={(e) => {
              const v = e.target.value;
              if (open) {
                setSearch(v);
              } else {
                setSearch(v);
                setInputValue(v);
                setOpen(true);
              }
            }}
            onFocus={() => {
              setOpen(true);
              setSearch(inputValue);
            }}
            disabled={disabled}
            className={cn(
              'w-full flex items-center gap-2 pl-4 pr-10 py-3 rounded-xl',
              'bg-secondary/30 border border-white/5',
              'placeholder:text-muted-foreground/60',
              'focus:outline-none focus:ring-2 focus:ring-primary/30 focus:bg-secondary/50',
              'text-sm'
            )}
          />
          <button
            type="button"
            onClick={() => !disabled && setOpen((o) => !o)}
            disabled={disabled}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded text-muted-foreground hover:text-foreground"
            aria-label="Toggle list"
          >
            <ChevronDown className={cn('w-4 h-4 transition-transform', showDropdown && 'rotate-180')} />
          </button>
        </div>

        {showDropdown && (
          <div className="absolute z-50 mt-1 w-full rounded-xl border border-white/10 bg-background/95 backdrop-blur-md shadow-xl overflow-hidden">
            <div className="max-h-72 overflow-y-auto p-1">
              <button
                type="button"
                onClick={() => handleSelect(null)}
                className="w-full py-2 px-3 rounded-lg text-left text-sm text-muted-foreground hover:bg-white/5"
              >
                Use sample data (no symbol file)
              </button>
              {loading ? (
                <div className="py-8 text-center text-sm text-muted-foreground">Loading symbols…</div>
              ) : !search.trim() ? (
                <div className="py-4 text-center text-xs text-muted-foreground">
                  Type above to search by security name or symbol
                </div>
              ) : filtered.length === 0 ? (
                <div className="py-6 text-center text-sm text-muted-foreground">No matches</div>
              ) : (
                filtered.map((s) => (
                  <button
                    key={s.symbol}
                    type="button"
                    onClick={() => handleSelect(s)}
                    className={cn(
                      'w-full py-2.5 px-3 rounded-lg text-left text-sm',
                      'hover:bg-white/5',
                      value?.symbol === s.symbol && 'bg-primary/10 text-primary'
                    )}
                  >
                    <span className="font-medium text-foreground block truncate">
                      {s.security_name}
                    </span>
                    <span className="text-muted-foreground text-xs flex items-center gap-1 mt-0.5">
                      Symbol: <span className="font-mono">{s.symbol}</span>
                      <span className="opacity-70">→ {fileHint(s)}</span>
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
