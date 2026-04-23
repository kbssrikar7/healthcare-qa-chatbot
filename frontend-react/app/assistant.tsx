"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useRuntimeFromContext } from "@/components/mediquery/chat-provider";
import { Thread } from "@/components/assistant-ui/thread";

export const Assistant = () => {
  const runtime = useRuntimeFromContext();

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh bg-zinc-950 text-zinc-100">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};
