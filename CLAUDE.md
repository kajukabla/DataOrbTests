# CLAUDE.md

## Debugging Web Apps

- **Never use screenshots to read logs or error messages.** Instead, build a logging server that captures `console.log`, `console.error`, and uncaught errors from the browser and streams them to the terminal where they can be read directly.
- When debugging browser-based projects, set up a WebSocket or HTTP-based log relay so all client-side output is available server-side.
