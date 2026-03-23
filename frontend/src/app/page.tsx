"use client";

import { useState } from "react";
import Hero from "@/components/Hero";
import ImageUpload from "@/components/ImageUpload";
import ResultView from "@/components/ResultView";
import TechStack from "@/components/TechStack";
import Footer from "@/components/Footer";

export default function Home() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [colorizedImage, setColorizedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processInfo, setProcessInfo] = useState<string | null>(null);
  const [samplingMethod, setSamplingMethod] = useState("ddim");
  const [quality, setQuality] = useState<"fast" | "balanced" | "best">("balanced");
  const qualityToSteps = { fast: 10, balanced: 20, best: 40 };

  const runColorization = async (file: File) => {

    setIsProcessing(true);
    setColorizedImage(null);
    setProcessInfo(null);
    setProgress(0);

    const formData = new FormData();
    formData.append("file", file);
    const totalSteps = qualityToSteps[quality];
    const startTime = Date.now();

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(
        `${apiUrl}/api/colorize-stream?steps=${totalSteps}`,
        { method: "POST", body: formData }
      );

      if (!res.ok) throw new Error("API error");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages (end with \n\n)
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || ""; // Keep incomplete part in buffer

        for (const part of parts) {
          const line = part.trim();
          if (line.startsWith("data: ") && line !== "data: [DONE]") {
            try {
              const data = JSON.parse(line.slice(6));
              setProgress((data.step / data.total) * 100);
              setColorizedImage(`data:image/png;base64,${data.image}`);
            } catch {
              // incomplete JSON, skip
            }
          }
        }
      }

      const elapsed = Date.now() - startTime;
      setProcessInfo(
        `Method: DDIM | Steps: ${totalSteps} | Time: ${elapsed}ms`
      );
    } catch {
      setProcessInfo(
        "Demo mode — API not connected. Deploy with a trained model to see live results."
      );
    } finally {
      setProgress(100);
      setTimeout(() => setIsProcessing(false), 300);
    }
  };

  const handleImageUpload = (file: File) => {
    setOriginalFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
    runColorization(file);
  };

  const handleRetry = () => {
    if (originalFile) {
      runColorization(originalFile);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setOriginalFile(null);
    setColorizedImage(null);
    setProcessInfo(null);
  };

  return (
    <main className="flex-1">
      <Hero />

      <section id="demo" className="max-w-5xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-3">Try It Out</h2>
          <p className="text-muted">
            Upload a grayscale or faded photo and watch AI bring it to life
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          <div className="glass rounded-xl px-4 py-3 flex items-center gap-3">
            <label className="text-sm text-muted">Method</label>
            <select
              value={samplingMethod}
              onChange={(e) => setSamplingMethod(e.target.value)}
              className="bg-card border border-card-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-accent [&>option]:bg-card [&>option]:text-foreground"
            >
              <option value="ddim">DDIM</option>
              <option value="piecewise">Piecewise</option>
              <option value="dpm_solver">DPM-Solver++</option>
            </select>
          </div>
          <div className="glass rounded-xl px-4 py-3 flex items-center gap-3">
            <label className="text-sm text-muted">Quality</label>
            <div className="bg-background/50 rounded-lg p-0.5 flex items-center gap-0">
              {(["fast", "balanced", "best"] as const).map((q) => (
                <button
                  key={q}
                  onClick={() => setQuality(q)}
                  className={`px-4 py-1.5 rounded-md text-sm capitalize transition-colors ${
                    quality === q
                      ? "bg-accent text-white"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {!originalImage ? (
          <ImageUpload onUpload={handleImageUpload} />
        ) : (
          <ResultView
            original={originalImage}
            colorized={colorizedImage}
            isProcessing={isProcessing}
            progress={progress}
            steps={qualityToSteps[quality]}
            processInfo={processInfo}
            onReset={handleReset}
            onRetry={handleRetry}
          />
        )}
      </section>

      <TechStack />
      <Footer />
    </main>
  );
}
