"use client";

import { motion } from "motion/react";
import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from "react-compare-slider";
import { RotateCcw, RefreshCw, Download, Loader2 } from "lucide-react";

interface Props {
  original: string;
  colorized: string | null;
  isProcessing: boolean;
  progress: number;
  steps: number;
  processInfo: string | null;
  onReset: () => void;
  onRetry: () => void;
}

export default function ResultView({
  original,
  colorized,
  isProcessing,
  progress,
  steps,
  processInfo,
  onReset,
  onRetry,
}: Props) {
  const handleDownload = () => {
    if (!colorized) return;
    const a = document.createElement("a");
    a.href = colorized;
    a.download = "kolorized.png";
    a.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Before/After Slider */}
      <div className="glass rounded-2xl overflow-hidden glow max-h-[70vh]">
        {isProcessing ? (
          <div className="relative">
            {/* Show live intermediate image if available */}
            {colorized ? (
              <img
                src={colorized}
                alt="In progress"
                className="w-full object-contain"
                style={{ maxHeight: "70vh" }}
              />
            ) : (
              <img
                src={original}
                alt="Processing"
                className="w-full object-contain opacity-40"
                style={{ maxHeight: "70vh" }}
              />
            )}
            {/* Overlay with progress */}
            <div className="absolute inset-0 flex flex-col items-center justify-end pb-8 bg-gradient-to-t from-black/70 via-transparent to-transparent">
              <div className="text-center w-full max-w-md px-6">
                <p className="text-lg font-semibold text-white mb-1">Colorizing...</p>
                <p className="text-sm text-white/70 mb-3">
                  Denoising step {Math.round(progress / 100 * steps)}/{steps}
                </p>
                <div className="w-full h-2 bg-white/20 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-accent rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>
            </div>
          </div>
        ) : colorized ? (
          <ReactCompareSlider
            itemOne={
              <ReactCompareSliderImage
                src={original}
                alt="Original grayscale"
                style={{ objectFit: "contain", maxHeight: "70vh" }}
              />
            }
            itemTwo={
              <ReactCompareSliderImage
                src={colorized}
                alt="Colorized"
                style={{ objectFit: "contain", maxHeight: "70vh" }}
              />
            }
            style={{ maxHeight: "70vh" }}
          />
        ) : (
          <div className="flex items-center justify-center" style={{ maxHeight: "70vh" }}>
            <img
              src={original}
              alt="Uploaded"
              className="max-w-full object-contain"
              style={{ maxHeight: "70vh" }}
            />
          </div>
        )}
      </div>

      {/* Labels */}
      {colorized && !isProcessing && (
        <div className="flex justify-between px-4 text-sm text-muted">
          <span>Original</span>
          <span>Kolorized</span>
        </div>
      )}

      {/* Info & Actions */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {processInfo && (
          <p className="text-sm text-muted font-mono">{processInfo}</p>
        )}

        <div className="flex gap-3 ml-auto">
          <button
            onClick={onRetry}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-accent/50 text-accent text-sm hover:bg-accent/10 transition-colors disabled:opacity-50"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-card-border text-sm hover:bg-card transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            New Image
          </button>
          {colorized && (
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-accent hover:bg-accent-hover text-white text-sm transition-colors"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
