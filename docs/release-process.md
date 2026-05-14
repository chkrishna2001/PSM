# Release Process

PSM Memory uses Changesets to maintain independent package versions and per-package release notes.

## Add a release note

After changing one or more public packages, run:

```bash
npm run changeset
```

Select the affected packages and version bump type:

- `patch`: bug fixes and small compatible changes
- `minor`: new compatible features
- `major`: breaking changes

Changesets creates a Markdown file in `.changeset/`. Commit that file with the code change.

## Version packages

When preparing a release, run:

```bash
npm run version-packages
```

This updates the affected package versions, internal workspace dependency ranges, and package changelogs.

## Publish packages

Publishing happens through GitHub Actions after a GitHub Release is published, or manually from the workflow dispatch screen.

Required GitHub secret:

```text
NPM_TOKEN
```

The public packages are:

- `@psm-memory/sdk`
- `@psm-memory/cli`
- `@psm-memory/pi-plugin`

The private benchmark package `@psm-memory/locomo` is ignored by Changesets and is not published.
