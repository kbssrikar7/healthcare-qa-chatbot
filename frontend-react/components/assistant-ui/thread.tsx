import {
  ComposerAddAttachment,
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";
import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  MoreHorizontalIcon,
  PencilIcon,
  RefreshCwIcon,
  SquareIcon,
} from "lucide-react";
import type { FC } from "react";
import { AnswerCard } from "@/components/mediquery/answer-card";
import { useMessageMetadata, useMediQuery } from "@/components/mediquery/chat-provider";
import { UI_SUGGESTIONS } from "@/components/mediquery/suggestions";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container flex h-full flex-col bg-zinc-950 text-zinc-100"
      style={{
        ["--thread-max-width" as string]: "min(96vw, 72rem)",
        ["--composer-radius" as string]: "20px",
        ["--composer-padding" as string]: "10px",
      }}
    >
      <ThreadPrimitive.Viewport
        turnAnchor="top"
        className="aui-thread-viewport relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll scroll-smooth px-4 pt-4"
      >
        {/* Empty state: welcome + composer centered in the full viewport */}
        <AuiIf condition={(s) => s.thread.isEmpty}>
          <ThreadWelcome />
        </AuiIf>

        <ThreadPrimitive.Messages>
          {() => <ThreadMessage />}
        </ThreadPrimitive.Messages>

        {/* Sticky bottom composer — only once messages exist */}
        <AuiIf condition={(s) => !s.thread.isEmpty}>
          <ThreadPrimitive.ViewportFooter className="aui-thread-viewport-footer sticky bottom-0 mx-auto mt-auto flex w-full max-w-(--thread-max-width) flex-col gap-4 overflow-visible rounded-t-(--composer-radius) bg-zinc-950 pb-4 md:pb-6">
            <ThreadScrollToBottom />
            <Composer compact />
          </ThreadPrimitive.ViewportFooter>
        </AuiIf>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  const isEditing = useAuiState((s) => s.message.composer.isEditing);
  if (isEditing) return <EditComposer />;
  if (role === "user") return <UserMessage />;
  return <AssistantMessage />;
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom render={<TooltipIconButton tooltip="Scroll to bottom" variant="outline" className="aui-thread-scroll-to-bottom absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible dark:border-border dark:bg-background dark:hover:bg-accent" />}><ArrowDownIcon /></ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root mx-auto flex w-full max-w-(--thread-max-width) min-h-full flex-col items-center justify-center pt-6 pb-16">
      {/* Branding */}
      <div className="aui-thread-welcome-message flex flex-col items-center gap-3 pb-5 px-4 text-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.svg" alt="MediQuery" width={52} height={52} className="mb-1" />
        <h1 className="aui-thread-welcome-message-inner animate-in fade-in slide-in-from-bottom-1 fill-mode-both font-semibold text-2xl duration-200">
          MediQuery
        </h1>
        <p className="aui-thread-welcome-message-inner animate-in fade-in slide-in-from-bottom-1 fill-mode-both text-muted-foreground text-base delay-75 duration-200 max-w-md">
          Ask about symptoms, conditions, treatments, or medications. Answers are grounded in verified medical literature with source attribution.
        </p>
      </div>

      {/* Composer */}
      <div className="w-full">
        <Composer />
      </div>

      {/* Suggestion chips — always below the input */}
      <SuggestionChips />
    </div>
  );
};

const SuggestionChips: FC = () => {
  const { openHistoryItem, isRunning } = useMediQuery();
  return (
    <div className="mx-auto flex w-full max-w-(--thread-max-width) flex-wrap justify-center gap-2 pt-3">
      {UI_SUGGESTIONS.map((chip) => (
        <button
          key={chip.prompt}
          type="button"
          disabled={isRunning}
          title={chip.description}
          onClick={() => openHistoryItem(chip.prompt)}
          className="inline-flex h-9 w-auto max-w-full items-center rounded-full border border-border bg-zinc-900/70 px-4 text-[13px] font-medium text-zinc-100 shadow-sm transition-colors duration-150 hover:border-ring/70 hover:bg-zinc-800/85 disabled:opacity-50"
        >
          <span className="truncate leading-tight">{chip.text}</span>
        </button>
      ))}
    </div>
  );
};

const Composer: FC<{ compact?: boolean }> = ({ compact = false }) => {
  return (
    <ComposerPrimitive.Root className="aui-composer-root relative flex w-full flex-col">
      <ComposerPrimitive.AttachmentDropzone render={<div data-slot="composer-shell" className={cn(
        "flex w-full flex-col rounded-(--composer-radius) border border-border bg-background transition-shadow focus-within:border-ring/75 focus-within:ring-2 focus-within:ring-ring/20 data-[dragging=true]:border-ring data-[dragging=true]:border-dashed data-[dragging=true]:bg-accent/50",
        compact ? "gap-0.5 p-2" : "gap-1 p-(--composer-padding)"
      )} />}><ComposerAttachments /><ComposerPrimitive.Input
                      placeholder="Ask a medical question..."
                      className={cn(
                        "aui-composer-input w-full resize-none bg-transparent outline-none placeholder:text-muted-foreground/80",
                        compact
                          ? "max-h-28 min-h-10 px-2 py-1.5 text-sm"
                          : "max-h-56 min-h-20 px-3 py-3 text-base"
                      )}
                      rows={1}
                      autoFocus
                      aria-label="Message input"
                    /><ComposerAction /></ComposerPrimitive.AttachmentDropzone>
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <div className="aui-composer-action-wrapper relative flex items-center justify-between">
      <ComposerAddAttachment />
      <AuiIf condition={(s) => !s.thread.isRunning}>
        <ComposerPrimitive.Send render={<TooltipIconButton tooltip="Send message" side="bottom" type="button" variant="default" size="icon" className="aui-composer-send size-8 rounded-full" aria-label="Send message" />}><ArrowUpIcon className="aui-composer-send-icon size-4" /></ComposerPrimitive.Send>
      </AuiIf>
      <AuiIf condition={(s) => s.thread.isRunning}>
        <ComposerPrimitive.Cancel render={<Button type="button" variant="default" size="icon" className="aui-composer-cancel size-8 rounded-full" aria-label="Stop generating" />}><SquareIcon className="aui-composer-cancel-icon size-3 fill-current" /></ComposerPrimitive.Cancel>
      </AuiIf>
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const AssistantMessage: FC = () => {
  const messageId = useAuiState((s) => s.message.id);
  const apiResponse = useMessageMetadata(messageId);

  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-3 duration-150"
      data-role="assistant"
    >
      <div className="aui-assistant-message-content wrap-break-word px-1 sm:px-0">
        <div className="overflow-hidden rounded-2xl border border-zinc-800/80 bg-zinc-900/40 shadow-sm ring-1 ring-zinc-800/40">
          <div className="px-4 py-3.5 text-[15px] leading-relaxed text-zinc-100 [&_a]:text-sky-400 [&_a]:underline-offset-2 [&_a]:hover:text-sky-300 [&_code]:rounded [&_code]:bg-zinc-800 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[13px] [&_li]:my-1 [&_ol]:my-2 [&_p]:my-2 [&_ul]:my-2">
            <MessagePrimitive.Parts>
              {({ part }) => {
                if (part.type === "text") return <MarkdownText />;
                if (part.type === "tool-call")
                  return part.toolUI ?? <ToolFallback {...part} />;
                return null;
              }}
            </MessagePrimitive.Parts>
            <MessageError />
          </div>

          {apiResponse && (
            <div className="border-t border-zinc-800/90 px-4 pb-4 pt-1">
              <AnswerCard response={apiResponse} />
            </div>
          )}
        </div>
      </div>

      <div className="aui-assistant-message-footer mt-2 flex min-h-8 items-center px-1">
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="aui-assistant-action-bar-root col-start-3 row-start-2 -ml-1 flex gap-1 text-muted-foreground"
    >
      <ActionBarPrimitive.Copy render={<TooltipIconButton tooltip="Copy" />}><AuiIf condition={(s) => s.message.isCopied}>
                      <CheckIcon />
                    </AuiIf><AuiIf condition={(s) => !s.message.isCopied}>
                      <CopyIcon />
                    </AuiIf></ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload render={<TooltipIconButton tooltip="Refresh" />}><RefreshCwIcon /></ActionBarPrimitive.Reload>
      <ActionBarMorePrimitive.Root>
        <ActionBarMorePrimitive.Trigger asChild><TooltipIconButton tooltip="More" className="data-[state=open]:bg-accent"><MoreHorizontalIcon /></TooltipIconButton></ActionBarMorePrimitive.Trigger>
        <ActionBarMorePrimitive.Content
          side="bottom"
          align="start"
          className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
        >
          <ActionBarPrimitive.ExportMarkdown render={<ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground" />}><DownloadIcon className="size-4" />Export as Markdown
                              </ActionBarPrimitive.ExportMarkdown>
        </ActionBarMorePrimitive.Content>
      </ActionBarMorePrimitive.Root>
    </ActionBarPrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-3 duration-150 [&:where(>*)]:col-start-2"
      data-role="user"
    >
      <UserMessageAttachments />

      <div className="aui-user-message-content-wrapper relative col-start-2 min-w-0">
        <div className="aui-user-message-content wrap-break-word peer rounded-2xl bg-zinc-800 px-4 py-2.5 text-zinc-100 empty:hidden">
          <MessagePrimitive.Parts />
        </div>
        <div className="aui-user-action-bar-wrapper absolute top-1/2 left-0 -translate-x-full -translate-y-1/2 pr-2 peer-empty:hidden">
          <UserActionBar />
        </div>
      </div>

      <BranchPicker className="aui-user-branch-picker col-span-full col-start-1 row-start-3 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="aui-user-action-bar-root flex flex-col items-end"
    >
      <ActionBarPrimitive.Edit render={<TooltipIconButton tooltip="Edit" className="aui-user-action-edit p-4" />}><PencilIcon /></ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-max-width) flex-col px-2 py-3">
      <ComposerPrimitive.Root className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl bg-zinc-800">
        <ComposerPrimitive.Input
          className="aui-edit-composer-input min-h-14 w-full resize-none bg-transparent p-4 text-foreground text-sm outline-none"
          autoFocus
        />
        <div className="aui-edit-composer-footer mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel render={<Button variant="ghost" size="sm" />}>Cancel
                              </ComposerPrimitive.Cancel>
          <ComposerPrimitive.Send render={<Button size="sm" />}>Update</ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "aui-branch-picker-root mr-2 -ml-2 inline-flex items-center text-muted-foreground text-xs",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous render={<TooltipIconButton tooltip="Previous" />}><ChevronLeftIcon /></BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next render={<TooltipIconButton tooltip="Next" />}><ChevronRightIcon /></BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
