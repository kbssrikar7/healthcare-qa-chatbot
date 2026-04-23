import { ChatProvider } from "@/components/mediquery/chat-provider";
import { SidebarLayout } from "@/components/mediquery/sidebar-layout";
import { Assistant } from "./assistant";

export default function Home() {
  return (
    <ChatProvider>
      <SidebarLayout>
        <Assistant />
      </SidebarLayout>
    </ChatProvider>
  );
}
