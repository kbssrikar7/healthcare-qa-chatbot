"use client";

import { useState } from "react";
import { PanelLeft } from "lucide-react";
import { Sidebar } from "./sidebar";
import { Button } from "@/components/ui/button";

export function SidebarLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex h-dvh overflow-hidden bg-zinc-950 text-zinc-100">

      {/* ── Sidebar panel ── */}
      <aside
        className={`
          relative flex flex-col shrink-0 border-r border-zinc-800 bg-zinc-900
          transition-[width] duration-300 ease-in-out overflow-hidden
          ${open ? "w-64" : "w-0"}
        `}
      >
        {/* Extra top padding so content clears the toggle button */}
        <div className="pt-2 flex-1 min-w-64 overflow-hidden">
          <Sidebar />
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className="relative flex flex-1 flex-col overflow-hidden">
        {/* Toggle button — pinned to top-left of viewport */}
        <div
          className={`fixed top-2 z-50 transition-[left] duration-300 ease-in-out ${
            open ? "left-[calc(16rem+0.5rem)]" : "left-2"
          }`}
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setOpen((o) => !o)}
            className="size-8 border border-zinc-700 bg-zinc-900 text-zinc-300 shadow-sm hover:bg-zinc-800 hover:text-zinc-100"
            aria-label={open ? "Close sidebar" : "Open sidebar"}
          >
            <PanelLeft className="size-4" />
          </Button>
        </div>

        {children}
      </div>
    </div>
  );
}
