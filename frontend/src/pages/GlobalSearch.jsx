// === Global Search ===
import React, { useState, useEffect } from "react";
import { jsonGET } from "../utils";

export function useDebouncedValue(value, delay = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

export function SearchBar({ value, onChange }) {
  return (
    <div className="w-full">
      <input
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search topics, entities, documents…"
        className="w-full rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm text-gray-900 placeholder-gray-500 shadow-sm transition-all outline-none focus:border-gray-400 focus:ring-2 focus:ring-gray-200 focus:ring-offset-0"
      />
    </div>
  );
}

export function SearchResults({ query, onPick }) {
  const deb = useDebouncedValue(query, 250);
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState([]);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancel = false;
    async function run() {
      if (!deb || deb.length < 2) {
        setItems([]);
        return;
      }
      setLoading(true);
      setErr(null);
      try {
        const data = await jsonGET(`/search?q=${encodeURIComponent(deb)}`);
        if (!cancel) setItems(data);
      } catch (e) {
        if (!cancel) setErr(String(e));
      } finally {
        if (!cancel) setLoading(false);
      }
    }
    run();
    return () => {
      cancel = true;
    };
  }, [deb]);

  const typeColors = {
    topic: "border-purple-200 bg-purple-50 text-purple-700",
    entity: "border-emerald-200 bg-emerald-50 text-emerald-700",
    document: "border-sky-200 bg-sky-50 text-sky-700",
  };

  return (
    <div className="mt-4 space-y-2">
      {loading && (
        <div className="px-2 py-2 text-sm text-gray-500">Searching…</div>
      )}
      {err && (
        <div className="px-2 py-2 text-sm text-red-600 bg-red-50 rounded-lg border border-red-200">
          {err}
        </div>
      )}
      {items.map((it) => {
        const t = it.type || "entity";
        const displayTitle = it.title || it.label || it.name || "(untitled)";
        const typeLabel = t[0].toUpperCase() + t.slice(1);
        return (
          <button
            key={`${t}-${it.id}`}
            onClick={() => onPick(it)}
            className="w-full rounded-lg border border-gray-200 bg-white px-4 py-3 text-left transition-all hover:border-gray-300 hover:bg-gray-50 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-200 focus:ring-offset-0"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="text-sm font-semibold text-gray-900 truncate flex-1">
                {displayTitle}
              </div>
              <span
                className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium flex-shrink-0 ${typeColors[t] || "border-gray-200 bg-gray-50 text-gray-600"
                  }`}
              >
                {typeLabel}
              </span>
            </div>
            {it.snippet && (
              <div className="mt-2 line-clamp-2 text-xs text-gray-600 leading-relaxed">
                {it.snippet}
              </div>
            )}
            {typeof it.score === "number" && (
              <div className="mt-2 text-xs text-gray-400 font-mono">
                Score: {it.score.toFixed(3)}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}