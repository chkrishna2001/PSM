# Company VM Install Hardening Plan

## Summary

Make `npm install -g @psm-memory/cli` a complete product install on restricted company VMs. A successful install means the user can run memory commands immediately after npm finishes. A CLI binary on PATH is not sufficient if config, DB, model, or embeddings are not ready.

Install must:

- install the CLI package.
- create config and DB in user-owned paths.
- install or verify the model.
- prepare embeddings.
- leave `psm-memory remember` and `psm-memory recall` usable.

If any of those steps fail, npm install should fail with concise terminal output and a detailed local install log.

## Current Failure Areas

- Global npm install may not have write access.
- `@psm-memory/cli` postinstall runs setup automatically, which is the desired product flow but currently needs better hardening.
- Setup may download the Qwen GGUF model during install, so install must handle network, proxy, certificate, and cache-write failures cleanly.
- Native dependencies can fail on locked-down Windows VMs, especially `better-sqlite3` and optional `node-llama-cpp`.
- Corporate antivirus or EDR tools may block, quarantine, or temporarily lock native binaries.

The observed `EPERM` install error means Windows denied a filesystem or process operation. For this package, likely denied operations are:

- writing to the global npm prefix, usually under `Program Files` or another admin-controlled location.
- extracting, replacing, or executing native dependency binaries, especially `.node` files.
- running native package install scripts for `better-sqlite3` or optional `node-llama-cpp`.
- postinstall setup writing config, model, DB, or cache files during npm install.
- antivirus or EDR locking a file while npm is trying to rename or delete it.

We need the exact denied path and npm lifecycle phase from logs before choosing the final fix.

## Clean Install Requirements

A clean install on any supported machine requires:

- A supported Node/npm version with available Windows binaries for required native dependencies.
- A writable npm installation prefix. If global admin paths are blocked, install must clearly report that and document a user-owned npm prefix path.
- Required native dependencies must either install from prebuilt binaries or have a supported fallback.
- Optional native dependencies must not fail the CLI install.
- Install-time setup must write only to user-owned PSM directories unless the user explicitly chooses another path.
- Model download, embedding initialization, config write, and DB write must succeed during `npm install`.
- Install-time setup must log every provisioning step and the exact failing path/operation when it fails.

## Install Modes

Support two documented install modes.

### Full Global Install

Use when global install is allowed:

```powershell
npm install -g @psm-memory/cli
psm-memory remember "Install smoke test memory."
psm-memory recall "What is the install smoke test memory?"
```

Expected behavior:

- CLI installs globally.
- Install creates config and memory DB.
- Install prepares model and embeddings.
- `psm-memory remember` and `psm-memory recall` work end to end immediately after install.

### Portable No-Global Install

Use when admin/global npm install is blocked:

```powershell
mkdir "$env:USERPROFILE\.psm-npm"
npm config set prefix "$env:USERPROFILE\.psm-npm"
$env:PATH="$env:USERPROFILE\.psm-npm;$env:PATH"
npm install -g @psm-memory/cli
psm-memory remember "Install smoke test memory."
psm-memory recall "What is the install smoke test memory?"
```

Expected behavior:

- install writes package files only to user-owned folders.
- no admin rights required.
- install-time setup and model preparation still happen.
- memory commands work immediately after install.

## Implementation Changes

### Reliable Postinstall

Make install-time setup reliable:

- `postinstall` should run non-interactive setup.
- `postinstall` should create config and DB in user-owned paths.
- `postinstall` should download or verify the model.
- `postinstall` should initialize or verify embeddings.
- `postinstall` should fail the install if memory is not usable.
- Failure output should be short and point to the detailed install log.

### Fix Native Dependency Install

Make npm dependency installation as reliable as possible:

- Keep `node-llama-cpp` optional so failure to install it does not fail CLI installation.
- Check whether `better-sqlite3` has Windows prebuild coverage for the supported Node version.
- If `better-sqlite3` requires local compilation on the target Node version, either pin to a version with prebuilds or move SQLite behind a supported fallback.
- Document the supported Node/npm versions for Windows.
- Treat `npm install -g @psm-memory/cli` failure as a package/dependency/install-time setup bug, not something to bypass with skip flags.

### Add Install Logs

Add explicit local logs so support can identify why install failed:

- During npm lifecycle scripts, write `install.log` under the PSM config/cache directory, or a temp fallback if the config directory cannot be created.
- Log Node version, npm version when available, platform, arch, npm prefix, package version, and environment flags relevant to proxy/certs.
- Log each step before and after it runs: package postinstall, config write, DB open, model download, embedding initialization, native runtime load.
- On failure, include the failing step, error code such as `EPERM`, syscall, path, destination path, and the suggested fix.
- Never log memory contents, prompts, model outputs, or secrets.

The install log is not a substitute for fixing the install path. It is how we determine whether the observed `EPERM` came from npm prefix permissions, native binaries, postinstall setup, model cache writes, or security software.

### Add Troubleshooting Documentation

Document common company VM failures:

- `EPERM` during npm install.
- missing Python/build tools for native modules.
- blocked global npm prefix.
- blocked model download.
- corporate proxy/certificate problems.
- antivirus/EDR locking `.node` native binaries.
- PowerShell execution policy.
- offline model placement if network model download is blocked.

## Acceptance Tests

A restricted Windows VM should be able to run:

```powershell
npm install -g @psm-memory/cli
psm-memory --help
psm-memory remember "Install smoke test memory."
psm-memory recall "What is the install smoke test memory?"
```

Success criteria:

- Install runs full setup and model preparation.
- CLI command is available.
- Memory commands work immediately after install.
- If install-time setup fails, npm install fails and the root cause is available in the install log.
- Any native dependency install failure during npm install is fixed by dependency/version/package changes, not bypassed with skip flags.

## Assumptions

- Windows company VM is the first target.
- Admin rights may be unavailable.
- GPU may be unavailable.
- Corporate network may block Hugging Face/model downloads.
- `npm install -g @psm-memory/cli` is the full product install and includes model provisioning.
