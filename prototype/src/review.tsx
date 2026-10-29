import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./review.css";

// Review tooling only. The iframe creates a real, independent CSS viewport.
const sizes = [
  [1440, 900],
  [1280, 800],
  [1024, 768],
  [1920, 1080],
] as const;
function Review() {
  const [size, setSize] = useState<readonly [number, number]>(sizes[0]);
  const host = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const observer = new ResizeObserver(() => {
      if (!host.current) return;
      setScale(
        Math.min(
          1,
          (host.current.clientWidth - 48) / size[0],
          (host.current.clientHeight - 48) / size[1],
        ),
      );
    });
    observer.observe(host.current!);
    return () => observer.disconnect();
  }, [size]);
  return (
    <div className="review-shell">
      <header>
        <strong>Viewport review</strong>
        <span>Isolated design prototype</span>
        <nav aria-label="Review viewport">
          {sizes.map(([w, h]) => (
            <button
              key={w}
              aria-pressed={size[0] === w}
              onClick={() => setSize([w, h])}
            >
              {w} × {h}
            </button>
          ))}
        </nav>
        <a href="/">Open prototype</a>
      </header>
      <div className="review-host" ref={host}>
        <div
          className="review-viewport"
          style={{ width: size[0] * scale, height: size[1] * scale }}
        >
          <iframe
            title="Boxing Motion design prototype"
            src="/"
            width={size[0]}
            height={size[1]}
            style={{ transform: `scale(${scale})` }}
          />
        </div>
      </div>
      <footer>
        Actual document viewport: {size[0]} × {size[1]} CSS pixels · Display
        scale: {Math.round(scale * 100)}%
      </footer>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<Review />);
