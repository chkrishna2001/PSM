# @psm-memory/pi-plugin

## 0.1.1

### Patch Changes

- 0123a14: Refine PSM Memory into a shared local memory product with grounded recall, interactive setup, and daemon autostart.

  - Add shared PSM-owned memory config, editable via `psm-memory config`.
  - Simplify the CLI to `remember "<text>"` and `recall "<question>"`.
  - Add interactive setup during global install with skip controls for CI.
  - Add daemon autostart with dynamic local port discovery through `daemon.json`.
  - Ground injected context in exact DB rows so recall cannot invent memory facts.
  - Keep Codex, Claude, and PI plugin integrations on the same shared memory store.

- Updated dependencies [0123a14]
  - @psm-memory/sdk@0.1.1

## 0.1.0

Initial PSM Memory PI plugin package.
