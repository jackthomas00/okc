// === Document View ===
import React from "react";
import { SectionCard } from "./components";

export default function DocumentView({ doc }) {
  if (!doc) return null;
  return (
    <SectionCard title="Document">
      <div className="text-xl font-semibold text-gray-900 mb-4">
        {doc.title || "(untitled document)"}
      </div>
      {doc.snippet && (
        <p className="text-sm text-gray-700 whitespace-pre-line leading-relaxed mb-4">
          {doc.snippet}
        </p>
      )}
      {doc.url && (
        <div className="pt-4 border-t border-gray-200">
          <a
            href={doc.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center text-sm font-medium text-blue-600 hover:text-blue-700 underline"
          >
            Open source document →
          </a>
        </div>
      )}
    </SectionCard>
  );
}