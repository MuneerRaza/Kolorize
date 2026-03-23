"use client";

import { motion } from "motion/react";
import { Brain, Zap, Layers, BarChart3 } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "Conditional DDPM",
    description:
      "Custom UNet with Lightweight Channel Attention and Gated Depthwise Conv FFN. v-prediction for faster convergence.",
  },
  {
    icon: Zap,
    title: "Fast Inference",
    description:
      "Multiple sampling strategies: DDIM, Piecewise skip sampling, and DPM-Solver++ for 12-step generation.",
  },
  {
    icon: Layers,
    title: "Min-SNR Optimization",
    description:
      "Min-SNR-γ loss weighting focuses training on informative timesteps. 3x faster convergence, same quality.",
  },
  {
    icon: BarChart3,
    title: "Production Ready",
    description:
      "ONNX export with FP16 quantization. FastAPI backend. Deployable on CPU with HuggingFace Spaces.",
  },
];

export default function TechStack() {
  return (
    <section className="max-w-5xl mx-auto px-6 py-20">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold mb-3">Architecture</h2>
        <p className="text-muted">
          Built with modern diffusion techniques for quality and speed
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feature, i) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.1 }}
            className="glass rounded-2xl p-6 hover:border-accent/30 transition-colors"
          >
            <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center mb-4">
              <feature.icon className="w-5 h-5 text-accent" />
            </div>
            <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
            <p className="text-sm text-muted leading-relaxed">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Pipeline visualization */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="mt-12 glass rounded-2xl p-8"
      >
        <h3 className="text-lg font-semibold mb-6 text-center">Pipeline</h3>
        <div className="flex flex-wrap items-center justify-center gap-4 text-sm">
          {[
            "Grayscale Input",
            "RGB → LAB",
            "L Channel (condition)",
            "DDPM Denoising",
            "Predicted AB",
            "LAB → RGB",
            "Color Output",
          ].map((step, i) => (
            <div key={step} className="flex items-center gap-4">
              <span className="px-4 py-2 rounded-xl bg-accent/10 text-accent border border-accent/20">
                {step}
              </span>
              {i < 6 && <span className="text-muted hidden sm:inline">→</span>}
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
