# AGENTS.md (frontend-react)

Guidance for AI coding agents working in the `frontend-react` Next.js app.

## Stack and Versions

- Next.js: `16.x`
- React: `19.x`
- TypeScript + App Router
- UI libraries: `@assistant-ui/*`, `@base-ui/react`, `radix-ui`, `lucide-react`

This project may include APIs and conventions newer than typical training data.
When uncertain, verify behavior against local docs in `node_modules/next/dist/docs/`.

## Development Commands

From `frontend-react/`:

```bash
npm install
npm run dev
npm run build
npm run lint
npm run start
```

Default local dev URL: `http://localhost:3000`.

## Project Structure

- `app/`: App Router pages, layouts, route handlers
- `app/api/chat/route.ts`: chat API route used by the assistant UI
- `components/assistant-ui/`: chat UI primitives and renderers
- `components/ui/`: shared UI building blocks
- `lib/`: shared utilities

## Working Guidelines

- Keep components small and composable.
- Prefer strict TypeScript types over `any`.
- Avoid introducing deprecated Next.js APIs.
- Preserve server/client component boundaries (`"use client"` only where needed).
- Keep styling consistent with existing utility patterns in the repo.

## Integration Notes

- This frontend is separate from the Streamlit UI in the repo root.
- If backend integration changes are needed, coordinate with root API contracts in `api/main.py`.
- Validate chat route behavior after changes to request/response payloads.
