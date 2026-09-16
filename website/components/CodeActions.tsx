"use client";
import { useEffect, useState } from "react";
export default function CodeActions() {
  const [message, setMessage] = useState("");
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function copy(event: MouseEvent) {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>(
        ".copy-code",
      );
      const code = button?.closest(".code-block")?.querySelector("code");
      if (!button || !code) return;
      try {
        await navigator.clipboard.writeText(code.textContent || "");
        button.textContent = "Copied";
        setMessage("Code copied to clipboard");
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
        button.textContent = "Selected";
        setMessage("Code selected. Use your browser copy command.");
      }
      timer = setTimeout(() => {
        button.textContent = "Copy";
        setMessage("");
      }, 2000);
    }
    document.addEventListener("click", copy);
    return () => {
      document.removeEventListener("click", copy);
      if (timer) clearTimeout(timer);
    };
  }, []);
  return (
    <span className="sr-only" role="status" aria-live="polite">
      {message}
    </span>
  );
}
