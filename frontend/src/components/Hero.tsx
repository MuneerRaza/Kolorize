"use client";

import { motion } from "motion/react";
import { Sparkles, ArrowDown } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 text-center max-w-3xl"
      >
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 mb-8"
        >
          <Sparkles className="w-4 h-4 text-accent" />
          <span className="text-sm text-muted">
            Powered by Conditional DDPM with v-prediction
          </span>
        </motion.div>

        {/* Title */}
        <h1 className="text-6xl md:text-8xl font-bold mb-6 tracking-tight">
          <span className="gradient-text">Kolorize</span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-muted mb-4 max-w-xl mx-auto">
          Bring old photos to life with AI-powered colorization
        </p>

        <p className="text-sm text-muted/60 mb-10 max-w-md mx-auto">
          Diffusion-based image colorization with Min-SNR optimization,
          multiple sampling strategies, and model compression for edge
          deployment.
        </p>

        {/* CTA */}
        <motion.a
          href="#demo"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="inline-flex items-center gap-2 bg-accent hover:bg-accent-hover text-white font-medium px-8 py-3 rounded-full transition-colors"
        >
          Try It Now
          <ArrowDown className="w-4 h-4" />
        </motion.a>

        {/* Tech pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="flex flex-wrap justify-center gap-2 mt-12"
        >
          {["DDPM", "v-prediction", "Min-SNR-γ", "DDIM", "DPM-Solver++", "ONNX", "FastAPI"].map(
            (tag) => (
              <span
                key={tag}
                className="text-xs px-3 py-1 rounded-full border border-card-border text-muted"
              >
                {tag}
              </span>
            )
          )}
        </motion.div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        animate={{ y: [0, 8, 0] }}
        transition={{ repeat: Infinity, duration: 2 }}
        className="absolute bottom-8"
      >
        <ArrowDown className="w-5 h-5 text-muted/40" />
      </motion.div>
    </section>
  );
}
