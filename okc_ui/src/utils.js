// === Imports ===
import { useState, useEffect } from "react";

// === Config ===
export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// --- Helpers ---
export async function jsonGET(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}