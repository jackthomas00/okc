import React, { useEffect, useState } from "react";
import { jsonGET } from "../utils";
import { Pill, SectionCard } from "./components";
import { Tag } from "./components";
import GraphView from "./GraphView";

export function EntityPage({ entityId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [activeRelType, setActiveRelType] = useState("all");

  useEffect(() => {
    let cancel = false;
    async function run() {
      if (!entityId) return;
      setLoading(true);
      setErr(null);
      try {
        const d = await jsonGET(`/entity/${entityId}`);
        if (!cancel) {
          setData(d);
          // Initialize tab
          const relTypes = Object.keys(d.relations_by_type || {});
          setActiveRelType(relTypes[0] || "all");
        }
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
  }, [entityId]);

  if (!entityId) {
    return (
      <div className="text-sm text-gray-500">
        Pick a topic/entity from search to start exploring.
      </div>
    );
  }
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-sm text-gray-500">Loading entity…</div>
      </div>
    );
  }
  if (err) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
        {err}
      </div>
    );
  }
  if (!data) return null;

  const relMap = data.relations_by_type || {};
  const relTypes = Object.keys(relMap);
  const evidence = data.evidence || [];

  const relEntries =
    activeRelType === "all"
      ? relTypes.flatMap((t) => relMap[t].map((e) => ({ ...e, _type: t })))
      : (relMap[activeRelType] || []).map((e) => ({ ...e, _type: activeRelType }));

  return (
    <div className="space-y-6">
      <SectionCard
        title="Entity"
        right={
          <div className="flex items-center gap-2">
            {data.type && <Pill>{data.type}</Pill>}
            <Tag>Entity</Tag>
          </div>
        }
      >
        <div className="text-xl font-semibold text-gray-900 mb-2">
          {data.name || data.canonical_label || data.id}
        </div>
        {data.canonical_label && data.canonical_label !== data.name && (
          <div className="mt-1 text-sm text-gray-500">
            Canonical: {data.canonical_label}
          </div>
        )}
        {data.definition && (
          <p className="mt-4 text-sm text-gray-700 whitespace-pre-line leading-relaxed">
            {data.definition}
          </p>
        )}
      </SectionCard>

      {/* Related entities */}
      {relTypes.length > 0 && (
        <SectionCard
          title="Related entities"
          right={
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => setActiveRelType("all")}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                  activeRelType === "all"
                    ? "bg-gray-900 text-white shadow-sm"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                All
              </button>
              {relTypes.map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveRelType(t)}
                  className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                    activeRelType === t
                      ? "bg-gray-900 text-white shadow-sm"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          }
        >
          {relEntries.length === 0 ? (
            <div className="py-4 text-sm text-gray-500 text-center">
              No related entities for this relation type.
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {relEntries.map((e) => (
                <div
                  key={e.id + e._type}
                  className="flex items-center justify-between rounded-lg border border-gray-200 bg-white px-4 py-3 shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="truncate flex-1 min-w-0">
                    <div className="font-semibold text-sm text-gray-900 truncate">
                      {e.name || e.canonical_label || e.id}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">{e._type}</div>
                  </div>
                  {typeof e.weight === "number" && (
                    <div className="ml-3 text-xs text-gray-500 font-mono flex-shrink-0">
                      w={e.weight.toFixed(2)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      )}

      {/* Graph */}
      <SectionCard title="Graph view">
        <GraphView centerId={entityId} kind="entity" />
      </SectionCard>

      {/* Evidence pane with provenance */}
      <SectionCard
        title="Evidence"
        right={
          <span className="text-xs text-gray-500">
            Provenance links go to source chunks/sentences.
          </span>
        }
      >
        {evidence.length === 0 && (
          <div className="py-4 text-sm text-gray-500 text-center">
            No evidence stored yet.
          </div>
        )}
        <div className="space-y-3">
          {evidence.map((ev) => (
            <div
              key={ev.id}
              className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3"
            >
              <div className="text-sm text-gray-800 leading-relaxed">{ev.text}</div>
              {ev.source && (
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-600">
                  <span className="font-medium">Source:</span>{" "}
                  {ev.source.url ? (
                    <a
                      href={ev.source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-600 hover:text-blue-700 underline"
                    >
                      {ev.source.document_title || "View document"}
                    </a>
                  ) : (
                    <span>{ev.source.document_title || "Unknown document"}</span>
                  )}
                  {ev.source.chunk_id != null && (
                    <> · chunk {ev.source.chunk_id}</>
                  )}
                  {ev.source.sentence_idx != null && (
                    <> · sentence {ev.source.sentence_idx}</>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}