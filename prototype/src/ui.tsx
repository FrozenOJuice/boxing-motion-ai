import {
  useEffect,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";

export type IconName =
  | "upload"
  | "play"
  | "pause"
  | "back"
  | "next"
  | "previous"
  | "arrow"
  | "close"
  | "expand"
  | "keyboard"
  | "info"
  | "check"
  | "file"
  | "chevron"
  | "bracket-in"
  | "bracket-out";
export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5" />
        <path d="M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />
      </>
    ),
    play: <path d="m9 5 11 7-11 7z" fill="currentColor" stroke="none" />,
    pause: (
      <>
        <path d="M8 5v14M16 5v14" strokeWidth="3" />
      </>
    ),
    back: <path d="m10 5-7 7 7 7M3 12h18" />,
    next: (
      <>
        <path d="m7 6 9 6-9 6zM20 5v14" />
      </>
    ),
    previous: (
      <>
        <path d="m17 6-9 6 9 6zM4 5v14" />
      </>
    ),
    arrow: <path d="M3 12h18m-7-7 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    expand: <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />,
    keyboard: (
      <>
        <rect x="2" y="5" width="20" height="14" rx="2" />
        <path d="M6 9h1m4 0h1m4 0h1M6 12h1m4 0h1m4 0h1M7 16h10" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6m0-10v.01" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    file: (
      <>
        <path d="M14 2H5v20h14V7l-5-5zM14 2v6h5" />
        <path d="m9 12 5 3-5 3z" />
      </>
    ),
    chevron: <path d="m9 5 7 7-7 7" />,
    "bracket-in": <path d="M16 4H8v16h8M3 12h10" />,
    "bracket-out": <path d="M8 4h8v16H8m13-8H11" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
export function Button({
  variant = "secondary",
  icon,
  children,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  icon?: IconName;
}) {
  return (
    <button {...props} className={`button button--${variant} ${className}`}>
      {icon && <Icon name={icon} />}
      {children}
    </button>
  );
}
export function IconButton({
  icon,
  label,
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  icon: IconName;
  label: string;
}) {
  return (
    <button
      {...props}
      aria-label={label}
      title={label}
      className={`icon-button ${props.className ?? ""}`}
    >
      <Icon name={icon} />
    </button>
  );
}
export function Dialog({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null),
    id = useId();
  useEffect(() => {
    const dialog = ref.current!;
    const previous =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    dialog.showModal();
    return () => {
      dialog.close();
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      aria-labelledby={id}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog-heading">
        <h2 id={id}>{title}</h2>
        <IconButton icon="close" label="Close dialog" onClick={onClose} />
      </div>
      {children}
    </dialog>
  );
}
export function Brand({ onClick }: { onClick: () => void }) {
  return (
    <button
      className="brand"
      onClick={onClick}
      aria-label="Boxing Motion AI — videos"
    >
      <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
        <rect width="28" height="28" rx="7" fill="currentColor" />
        <path
          d="m8 19 4-10 3 6 3-6 3 10"
          stroke="white"
          strokeWidth="1.7"
          fill="none"
          strokeLinejoin="round"
        />
      </svg>
      <span>
        Boxing Motion <span className="brand-ai">AI</span>
      </span>
    </button>
  );
}
