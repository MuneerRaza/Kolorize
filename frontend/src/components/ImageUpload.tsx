"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "motion/react";
import { Upload, ImageIcon } from "lucide-react";

interface Props {
  onUpload: (file: File) => void;
}

export default function ImageUpload({ onUpload }: Props) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0]);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp", ".bmp"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div
        {...getRootProps()}
        className={`upload-zone rounded-2xl p-16 text-center cursor-pointer transition-all ${
          isDragActive ? "border-accent bg-accent/5" : "bg-card/50"
        }`}
      >
        <input {...getInputProps()} />

        <motion.div
          animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="w-16 h-16 rounded-2xl bg-accent/10 flex items-center justify-center">
            {isDragActive ? (
              <ImageIcon className="w-8 h-8 text-accent" />
            ) : (
              <Upload className="w-8 h-8 text-accent" />
            )}
          </div>

          <div>
            <p className="text-lg font-medium mb-1">
              {isDragActive
                ? "Drop your image here"
                : "Drag & drop an image, or click to browse"}
            </p>
            <p className="text-sm text-muted">
              Supports PNG, JPG, WebP up to 10MB
            </p>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
