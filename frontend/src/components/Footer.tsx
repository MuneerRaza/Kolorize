import { Github } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-card-border py-8 px-6">
      <div className="max-w-5xl mx-auto flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="font-semibold">Kolorize</p>
          <p className="text-sm text-muted">
            Diffusion-based image colorization by{" "}
            <a
              href="https://github.com/MuneerRaza"
              className="text-accent hover:text-accent-hover transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Muneer Raza
            </a>
          </p>
        </div>
        <a
          href="https://github.com/MuneerRaza/Kolorize"
          className="flex items-center gap-2 text-sm text-muted hover:text-foreground transition-colors"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Github className="w-5 h-5" />
          View Source
        </a>
      </div>
    </footer>
  );
}
