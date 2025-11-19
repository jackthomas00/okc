// === UI Primitives ===
export function Pill({ children, variant = "default" }) {
  const base =
    "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors";
  const styles =
    variant === "solid"
      ? "bg-slate-900 text-white border-slate-900"
      : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50";
  return <span className={`${base} ${styles}`}>{children}</span>;
}

export function Tag({ children }) {
  return (
    <span className="inline-flex items-center rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
      {children}
    </span>
  );
}

export function SectionCard({ title, children, right }) {
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {right && <div className="flex items-center gap-2">{right}</div>}
      </div>
      <div>{children}</div>
    </section>
  );
}


