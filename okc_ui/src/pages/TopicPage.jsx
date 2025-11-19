// === Topic Page ===
import React, { useEffect, useState } from "react";
import { jsonGET } from "../utils";
import { Pill, SectionCard } from "./components";
import GraphView from "./GraphView";

export function TopicPage({ topicId, onSelectEntity }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancel = false;
    async function run() {
      if (!topicId) return;
      setLoading(true);
      setErr(null);
      try {
        const d = await jsonGET(`/topic/${topicId}`);
        if (!cancel) setData(d);
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
  }, [topicId]);

  if (!topicId) return null;
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-sm text-gray-500">Loading topic…</div>
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

  const members = data.members || [];

  return (
    <div className="space-y-6">
      <SectionCard
        title="Topic"
        right={data.label && <Pill variant="solid">Topic</Pill>}
      >
        <div className="text-xl font-semibold text-gray-900 mb-3">
          {data.label || data.name || "(untitled topic)"}
        </div>
        {data.summary && (
          <p className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
            {data.summary}
          </p>
        )}
      </SectionCard>

      {members.length > 0 && (
        <SectionCard title="Key entities">
          <div className="flex flex-wrap gap-2.5">
            {members.map((e) => (
              <button
                key={e.id}
                onClick={() => onSelectEntity(e)}
                className="rounded-full border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-800 shadow-sm transition-all hover:border-gray-400 hover:bg-gray-50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-gray-200 focus:ring-offset-0"
              >
                {e.name || e.canonical_label || e.id}
              </button>
            ))}
          </div>
        </SectionCard>
      )}

      <SectionCard title="Mini graph">
        <GraphView centerId={topicId} kind="topic" />
      </SectionCard>
    </div>
  );
}