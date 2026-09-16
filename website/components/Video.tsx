"use client";
import { useEffect, useRef, useState } from "react";
import { asset } from "@/lib/config";
export default function Video({
  file,
  title,
  autoplay = false,
}: {
  file: string;
  title: string;
  autoplay?: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const video = ref.current;
    if (!video || !autoplay) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    const motion = () => {
      if (reduced.matches) video.pause();
    };
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !reduced.matches)
          video.play().catch(() => {});
        else video.pause();
      },
      { threshold: 0.25 },
    );
    observer.observe(video);
    reduced.addEventListener("change", motion);
    return () => {
      observer.disconnect();
      reduced.removeEventListener("change", motion);
      video.pause();
    };
  }, [autoplay, file]);
  return (
    <div className="overflow-hidden rounded-xl bg-[#0b1220]">
      <video
        ref={ref}
        key={file}
        controls
        muted
        loop={autoplay}
        playsInline
        preload={autoplay ? "metadata" : "none"}
        poster={asset(`media/${file}.jpg`)}
        aria-label={title}
        className="aspect-video w-full"
        onError={() => setError(true)}
      >
        <source src={asset(`media/${file}.mp4`)} type="video/mp4" />
        Your browser cannot play this video.{" "}
        <a href={asset(`media/${file}.mp4`)}>Download the MP4.</a>
      </video>
      {error && (
        <p role="status" className="p-4 text-sm text-white">
          The video could not load.{" "}
          <a className="underline" href={asset(`media/${file}.mp4`)}>
            Open the MP4 directly
          </a>
          .
        </p>
      )}
    </div>
  );
}
