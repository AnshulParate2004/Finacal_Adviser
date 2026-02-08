import { motion } from 'framer-motion';

export function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center p-12 space-y-8 absolute inset-0">
      <div className="relative w-32 h-32">
        {/* Outer rotating ring */}
        <motion.div
          className="absolute inset-0 border-4 border-transparent border-t-primary rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />

        {/* Middle rotating ring (reverse) */}
        <motion.div
          className="absolute inset-4 border-4 border-transparent border-t-accent rounded-full opacity-70"
          animate={{ rotate: -360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        {/* Inner rotating ring */}
        <motion.div
          className="absolute inset-8 border-4 border-transparent border-t-success rounded-full opacity-50"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />

        {/* Center pulsing core */}
        <motion.div
          className="absolute inset-0 m-auto w-4 h-4 bg-white rounded-full glow-box"
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </div>

      <div className="text-center space-y-2">
        <motion.h3
          className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          Processing Strategy
        </motion.h3>
        <div className="flex flex-col gap-1">
          <p className="text-muted-foreground text-sm">Analyzing market patterns...</p>
          <motion.div
            className="h-1 bg-gradient-to-r from-primary to-accent rounded-full mx-auto"
            initial={{ width: 0 }}
            animate={{ width: 100 }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </div>
      </div>
    </div>
  );
}
