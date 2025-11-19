// === Graph View (with filters & edge evidence modal) ===
import React, { useEffect, useMemo, useRef, useState } from "react";
import { jsonGET } from "../utils";
import { Tag } from "./components";

export function EdgeEvidenceModal({ edge, onClose }) {
  if (!edge) return null;
  const evidence = edge.evidence || [];
  return (
    <div 
      className="fixed inset-0 z-30 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div 
        className="max-h-[85vh] w-full max-w-2xl overflow-hidden rounded-xl bg-white shadow-2xl border border-gray-200"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 bg-gray-50">
          <div className="text-base font-semibold text-gray-900">
            Evidence for edge: {edge.type || "relation"}
          </div>
          <button
            className="rounded-lg px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-200 transition-colors font-medium"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <div className="max-h-[75vh] space-y-4 overflow-auto p-6">
          <div className="flex flex-wrap gap-2">
            {edge.type && <Tag>Type: {edge.type}</Tag>}
            {typeof edge.confidence === "number" && (
              <Tag>Confidence: {edge.confidence.toFixed(2)}</Tag>
            )}
            {edge.weight != null && <Tag>Weight: {edge.weight}</Tag>}
          </div>
          {evidence.length === 0 && (
            <div className="py-8 text-sm text-gray-500 text-center">
              No evidence attached.
            </div>
          )}
          {evidence.map((ev) => (
            <div
              key={ev.id}
              className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3"
            >
              <div className="text-sm text-gray-800 leading-relaxed">{ev.text}</div>
              {ev.source && (
                <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-600">
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
      </div>
    </div>
  );
}


export default function GraphView({ centerId, kind }) {
  const containerRef = useRef(null);
  const cyInstanceRef = useRef(null);
  const resizeHandlerRef = useRef(null);
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const [minConf, setMinConf] = useState(0);
  const [selectedTypes, setSelectedTypes] = useState(() => new Set());
  const [edgeForModal, setEdgeForModal] = useState(null);

  // Fetch graph
  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!centerId) return;
      setLoading(true);
      setErr(null);
      try {
        const path =
          kind === "topic"
            ? `/graph/topic/${centerId}`
            : `/graph/entity/${centerId}`;
        const data = await jsonGET(path);
        if (!cancelled) {
          setGraph(data);
          setEdgeForModal(null);
        }
      } catch (e) {
        if (!cancelled) setErr(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [centerId, kind]);

  // Distinct edge types - infer default type from kind if not provided
  const allTypes = useMemo(() => {
    if (!graph?.edges) return [];
    const defaultType = kind === "topic" ? "similar" : "co-mention";
    const set = new Set(
      graph.edges.map((e) => e.type || defaultType).filter(Boolean)
    );
    return Array.from(set);
  }, [graph, kind]);

  // Filtered edges
  const filteredEdges = useMemo(() => {
    if (!graph?.edges) return [];
    const defaultType = kind === "topic" ? "similar" : "co-mention";
    return graph.edges.filter((e) => {
      const edgeType = e.type || defaultType;
      const okConf =
        typeof e.confidence === "number" ? e.confidence >= minConf : true;
      const okType =
        selectedTypes.size === 0 || selectedTypes.has(edgeType);
      return okConf && okType;
    });
  }, [graph, minConf, selectedTypes, kind]);

  // Build nodes from filtered edges plus center
  const filteredNodes = useMemo(() => {
    if (!graph?.nodes) return [];
    const allowedIds = new Set();
    // Add nodes referenced by filtered edges
    filteredEdges.forEach((e) => {
      allowedIds.add(e.source);
      allowedIds.add(e.target);
    });
    // Format centerId to match node ID format (t:{id} for topics, e:{id} for entities)
    if (centerId) {
      const formattedCenterId = kind === "topic" ? `t:${centerId}` : `e:${centerId}`;
      allowedIds.add(formattedCenterId);
      // Also add the raw centerId in case the API uses a different format
      allowedIds.add(String(centerId));
    }
    // If no edges but we have a centerId, ensure we include at least the center node
    const nodes = graph.nodes.filter((n) => allowedIds.has(n.id));
    // If we have no nodes but should have a center, try to find it
    if (nodes.length === 0 && centerId && graph.nodes.length > 0) {
      const formattedCenterId = kind === "topic" ? `t:${centerId}` : `e:${centerId}`;
      const centerNode = graph.nodes.find((n) => n.id === formattedCenterId || n.id === String(centerId));
      if (centerNode) return [centerNode];
    }
    return nodes;
  }, [graph, filteredEdges, centerId, kind]);

  // Render graph with Cytoscape
  useEffect(() => {
    if (!containerRef.current || !graph) return;
    let cancelled = false;
    let resizeObserver = null;

    // Cleanup previous instance
    if (cyInstanceRef.current) {
      if (resizeHandlerRef.current) {
        window.removeEventListener("resize", resizeHandlerRef.current);
      }
      cyInstanceRef.current.destroy();
      cyInstanceRef.current = null;
      resizeHandlerRef.current = null;
    }

    const initializeCytoscape = (cytoscape) => {
      if (cancelled || !containerRef.current) return;

      const container = containerRef.current;
      
      // Check if container has dimensions
      if (container.offsetHeight === 0 || container.offsetWidth === 0) {
        return false; // Indicate we need to wait
      }

      const defaultType = kind === "topic" ? "similar" : "co-mention";
      
      // Helper function to truncate labels
      const truncateLabel = (text, maxLength = 20) => {
        if (!text) return "";
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + "...";
      };
      
      const elements = [
        ...filteredNodes.map((n) => ({
          data: {
            id: n.id,
            label: truncateLabel(n.label || n.name || n.canonical_label || n.id),
          },
        })),
        ...filteredEdges.map((e, idx) => ({
          data: {
            id: e.id || `edge-${idx}`,
            source: e.source,
            target: e.target,
            type: e.type || defaultType,
            weight: e.weight,
            confidence: e.confidence,
            evidence: e.evidence,
          },
        })),
      ];

      if (elements.length === 0) {
        console.warn("GraphView: No elements to render");
        return true;
      }

      const cy = cytoscape({
        container: container,
        elements,
        userPanningEnabled: true,
        userZoomingEnabled: true,
        boxSelectionEnabled: false,
        minZoom: 0.1,
        maxZoom: 4,
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "text-valign": "center",
              "text-halign": "center",
              "font-size": 11,
              "background-color": "#0f172a",
              color: "#f9fafb",
              width: 80,
              height: 40,
              "shape": "round-rectangle",
              "padding": "4px",
              "text-wrap": "wrap",
              "text-max-width": "70px",
            },
          },
          {
            selector: "edge",
            style: {
              width: 1.5,
              "line-color": "#cbd5e1",
              "target-arrow-color": "#cbd5e1",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
            },
          },
        ],
      });

      // Run layout explicitly
      const layout = cy.layout({ name: "cose", animate: false });
      layout.run();

      // Ensure container is properly sized
      cy.resize();

      cy.on("tap", "edge", (evt) => {
        const d = evt.target.data();
        setEdgeForModal({
          id: d.id,
          type: d.type,
          confidence: d.confidence,
          weight: d.weight,
          evidence: d.evidence || [],
        });
      });

      const resize = () => cy.resize();
      window.addEventListener("resize", resize);
      cyInstanceRef.current = cy;
      resizeHandlerRef.current = resize;

      return true; // Success
    };

    import("cytoscape")
      .then(({ default: cytoscape }) => {
        if (cancelled || !containerRef.current) return;

        // Try to initialize immediately
        if (initializeCytoscape(cytoscape)) {
          return; // Success
        }

        // If container has no dimensions, use ResizeObserver to wait for it
        const container = containerRef.current;
        resizeObserver = new ResizeObserver((entries) => {
          if (cancelled) return;
          for (const entry of entries) {
            if (entry.contentRect.height > 0 && entry.contentRect.width > 0) {
              if (!cyInstanceRef.current) {
                initializeCytoscape(cytoscape);
              }
              resizeObserver?.disconnect();
              resizeObserver = null;
              break;
            }
          }
        });
        resizeObserver.observe(container);
      })
      .catch((error) => {
        console.error("Failed to load Cytoscape:", error);
        if (!cancelled) {
          setErr(`Failed to load graph library: ${error.message}`);
        }
      });

    return () => {
      cancelled = true;
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      if (cyInstanceRef.current) {
        if (resizeHandlerRef.current) {
          window.removeEventListener("resize", resizeHandlerRef.current);
        }
        cyInstanceRef.current.destroy();
        cyInstanceRef.current = null;
        resizeHandlerRef.current = null;
      }
    };
  }, [graph, filteredNodes, filteredEdges, kind]);

  if (!centerId) {
    return (
      <div className="flex h-80 items-center justify-center rounded-xl border border-gray-200 bg-gray-50 text-sm text-gray-500">
        Center on a topic or entity to view its graph.
      </div>
    );
  }
  if (loading) {
    return (
      <div className="flex h-80 items-center justify-center rounded-xl border border-gray-200 bg-gray-50 text-sm text-gray-500">
        Loading graph…
      </div>
    );
  }
  if (err) {
    return (
      <div className="flex h-80 items-center justify-center rounded-xl border border-red-200 bg-red-50 text-sm text-red-600">
        {err}
      </div>
    );
  }

  return (
    <>
      <div className="mb-4 flex flex-col gap-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <div className="flex items-center gap-4">
          <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
            Min confidence:
          </label>
          <div className="flex items-center gap-3 flex-1">
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={minConf}
              onChange={(e) => setMinConf(parseFloat(e.target.value))}
              className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-gray-700"
            />
            <span className="w-12 text-right tabular-nums text-sm font-medium text-gray-900">
              {minConf.toFixed(1)}
            </span>
          </div>
        </div>
        {allTypes.length > 0 && (
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-gray-700 whitespace-nowrap">Edge types:</span>
            <div className="flex flex-wrap items-center gap-2 flex-1">
              {allTypes.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer hover:text-gray-900">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-gray-700 focus:ring-gray-200 cursor-pointer"
                    checked={selectedTypes.has(t)}
                    onChange={(e) => {
                      setSelectedTypes((prev) => {
                        const next = new Set(prev);
                        if (e.target.checked) next.add(t);
                        else next.delete(t);
                        return next;
                      });
                    }}
                  />
                  <span className="text-gray-700">{t}</span>
                </label>
              ))}
              <button
                className="ml-auto rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                onClick={() => setSelectedTypes(new Set())}
              >
                Clear all
              </button>
            </div>
          </div>
        )}
      </div>
      <div
        ref={containerRef}
        className="h-96 w-full rounded-xl border border-gray-200 bg-white shadow-sm"
        style={{ minHeight: '384px', height: '384px', touchAction: 'none' }}
      />
      <EdgeEvidenceModal edge={edgeForModal} onClose={() => setEdgeForModal(null)} />
    </>
  );
}