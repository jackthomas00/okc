import React, { useState } from "react";
import { SearchBar, SearchResults } from "./pages/GlobalSearch";
import { TopicPage } from "./pages/TopicPage";
import { EntityPage } from "./pages/EntityPage";
import DocumentView from "./pages/DocumentView";
import { API_BASE } from "./utils";

export default function App() {
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState(null); // { type, id, ... }

  const pickedType = picked?.type || "entity";

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-8 border-b border-gray-200 pb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Open Knowledge Compiler — Explorer
          </h1>
          <p className="text-sm text-gray-600">
            Global search → topic/entity pages → graph + evidence
          </p>
          <p className="mt-1 text-xs text-gray-500 font-mono">
            API: {API_BASE}
          </p>
        </div>
      </header>

      <div className="grid gap-8 md:grid-cols-12">
        {/* Left: Global search */}
        <aside className="md:col-span-4 lg:col-span-3">
          <SearchBar value={q} onChange={setQ} />
          <SearchResults
            query={q}
            onPick={(it) => {
              setPicked(it);
              const displayTitle = it.title || it.label || it.name || "";
              if (displayTitle) setQ(displayTitle);
            }}
          />
        </aside>

        {/* Right: main content */}
        <main className="md:col-span-8 lg:col-span-9 space-y-6">
          {!picked && (
            <div className="rounded-xl border border-gray-200 bg-gradient-to-br from-gray-50 to-white p-8">
              <div className="mb-4 text-xl font-semibold text-gray-900">
                Welcome to the OKC graph explorer
              </div>
              <p className="mb-4 text-sm text-gray-700 leading-relaxed">
                Start by searching for a topic, entity, or document. You'll see:
              </p>
              <ul className="space-y-3 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="mr-3 text-gray-400">•</span>
                  <span>
                    <strong className="font-semibold text-gray-900">Topic pages</strong> with labels, summaries, key entities, and mini-graph
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-3 text-gray-400">•</span>
                  <span>
                    <strong className="font-semibold text-gray-900">Entity pages</strong> with definitions, relation-type tabs, and an evidence pane
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-3 text-gray-400">•</span>
                  <span>
                    <strong className="font-semibold text-gray-900">Graph views</strong> where you can filter edges and click them to see supporting evidence
                  </span>
                </li>
              </ul>
            </div>
          )}

          {picked && pickedType === "topic" && (
            <TopicPage
              topicId={picked.id}
              onSelectEntity={(e) =>
                setPicked({ ...e, type: "entity" })
              }
            />
          )}

          {picked && pickedType === "entity" && (
            <EntityPage entityId={picked.id} />
          )}

          {picked && pickedType === "document" && <DocumentView doc={picked} />}
        </main>
      </div>

      <footer className="mt-12 pt-6 border-t border-gray-200 text-xs text-gray-500">
        Built for OKC v0.x — expected endpoints:{" "}
        <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">/search</code>,{" "}
        <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">/topic/&lt;id&gt;</code>,{" "}
        <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">/entity/&lt;id&gt;</code>,{" "}
        <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">/graph/topic/&lt;id&gt;</code>,{" "}
        <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700">/graph/entity/&lt;id&gt;</code>.
      </footer>
    </div>
  );
}
