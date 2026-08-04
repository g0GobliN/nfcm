import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { getSnapshot } from "./api";
import type { RuntimeSnapshot } from "./types";

interface RuntimeContextValue {
  snapshot: RuntimeSnapshot | null;
  error: string | null;
  refresh: () => Promise<void>;
  setSnapshot: (s: RuntimeSnapshot) => void;
}

const RuntimeContext = createContext<RuntimeContextValue | null>(null);

export function RuntimeProvider({ children }: { children: ReactNode }) {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const snap = await getSnapshot();
      setSnapshot(snap);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  return (
    <RuntimeContext.Provider value={{ snapshot, error, refresh, setSnapshot }}>
      {children}
    </RuntimeContext.Provider>
  );
}

export function useRuntime() {
  const ctx = useContext(RuntimeContext);
  if (!ctx) throw new Error("useRuntime outside provider");
  return ctx;
}
