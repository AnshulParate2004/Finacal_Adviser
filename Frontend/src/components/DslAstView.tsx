import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { ChevronRight, ChevronDown, Code2, Binary } from 'lucide-react';
import type { ASTNodeTree } from '@/types/strategy';

interface DslAstViewProps {
  dslCode?: string;
  astTree?: ASTNodeTree;
}

/** Single AST node in the tree */
function AstTreeNode({ node, depth = 0 }: { node: ASTNodeTree; depth?: number }) {
  const [open, setOpen] = useState(depth < 2);
  const hasChildren =
    node.entry !== undefined ||
    node.exit !== undefined ||
    node.left !== undefined ||
    node.right !== undefined ||
    (node as { series?: ASTNodeTree }).series !== undefined;

  const nodeType = (node as ASTNodeTree & { node_type?: string }).type ?? (node as ASTNodeTree & { node_type?: string }).node_type ?? 'unknown';
  const label = (() => {
    if (nodeType === 'strategy') return 'Strategy';
    if (nodeType === 'comparison') return `Comparison (${node.operator})`;
    if (nodeType === 'boolean_op') return `BooleanOp (${node.operator})`;
    if (nodeType === 'series') return `Series: ${node.name ?? (typeof (node as { series?: string }).series === 'string' ? (node as { series?: string }).series : null) ?? '?'}`;
    if (nodeType === 'number') return `Number: ${node.value}`;
    if (nodeType === 'indicator') return `Indicator: ${node.name}(${node.params?.join(', ') ?? ''})`;
    if (nodeType === 'time_reference') return `TimeRef: ${node.series}_${node.lag}`;
    if (nodeType === 'scale') return `Scale (× ${node.multiplier})`;
    return nodeType;
  })();

  const seriesNode = (node as { series?: ASTNodeTree }).series;

  return (
    <div className="select-none">
      <div
        className="flex items-center gap-1 py-0.5 rounded px-1 hover:bg-white/5 cursor-pointer min-h-7"
        style={{ paddingLeft: `${depth * 12 + 4}px` }}
        onClick={() => hasChildren && setOpen((o) => !o)}
      >
        {hasChildren ? (
          open ? (
            <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
          ) : (
            <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
          )
        ) : (
          <span className="w-4 shrink-0" />
        )}
        <span className="text-xs font-medium text-accent">{nodeType}</span>
        <span className="text-xs text-muted-foreground truncate"> — {label}</span>
      </div>
      <AnimatePresence>
        {open && hasChildren && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {node.entry && <AstTreeNode node={node.entry} depth={depth + 1} />}
            {node.exit && <AstTreeNode node={node.exit} depth={depth + 1} />}
            {node.left && <AstTreeNode node={node.left} depth={depth + 1} />}
            {node.right && <AstTreeNode node={node.right} depth={depth + 1} />}
            {seriesNode && typeof seriesNode === 'object' && (
              <AstTreeNode node={seriesNode} depth={depth + 1} />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function DslAstView({ dslCode, astTree }: DslAstViewProps) {
  const hasDsl = Boolean(dslCode?.trim());
  const hasAst = Boolean(astTree);
  const [activeTab, setActiveTab] = useState<'dsl' | 'ast'>(() =>
    hasDsl ? 'dsl' : 'ast'
  );

  if (!hasDsl && !hasAst) return null;

  return (
    <Card className="p-6 glass-card flex flex-col relative overflow-hidden">
      <div className="absolute -bottom-20 -right-20 w-48 h-48 bg-accent/5 rounded-full blur-3xl" />

      <div className="flex items-center gap-2 mb-4 relative z-10">
        <Code2 className="w-5 h-5 text-accent" />
        <span className="text-lg font-semibold">DSL & AST</span>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4 relative z-10">
        {hasDsl && (
          <button
            type="button"
            onClick={() => setActiveTab('dsl')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'dsl'
                ? 'bg-primary/20 text-primary border border-primary/30'
                : 'bg-white/5 text-muted-foreground border border-transparent hover:bg-white/10'
            }`}
          >
            <Code2 className="w-4 h-4" />
            DSL
          </button>
        )}
        {hasAst && (
          <button
            type="button"
            onClick={() => setActiveTab('ast')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'ast'
                ? 'bg-primary/20 text-primary border border-primary/30'
                : 'bg-white/5 text-muted-foreground border border-transparent hover:bg-white/10'
            }`}
          >
            <Binary className="w-4 h-4" />
            AST Tree
          </button>
        )}
      </div>

      <div className="relative z-10 flex-1 min-h-[120px]">
        <AnimatePresence mode="wait">
          {activeTab === 'dsl' && hasDsl && (
            <motion.div
              key="dsl"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-lg bg-black/30 border border-white/10 p-4 font-mono text-sm overflow-x-auto"
            >
              <pre className="whitespace-pre text-muted-foreground">
                {dslCode}
              </pre>
            </motion.div>
          )}
          {activeTab === 'ast' && hasAst && astTree && (
            <motion.div
              key="ast"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-lg bg-black/30 border border-white/10 p-3 overflow-auto max-h-[320px]"
            >
              <AstTreeNode node={astTree} depth={0} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Card>
  );
}
