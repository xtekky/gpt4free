# gpt4free documentation

The public documentation is served from the **g4f.dev** checkout at
<https://g4f.dev/docs/>. The Markdown files in this directory are the source of
truth; `bin/publish-docs` copies them into `../g4f.dev/docs` and generates the
matching `.html` pages with the website's `docs/template.html`.

## Publishing

```bash
# Check whether the website checkout is up to date
bin/publish-docs --check

# Copy Markdown and render HTML into ../g4f.dev/docs
bin/publish-docs

# Remove pages that were published earlier but no longer exist here
bin/publish-docs --clean
```

The publisher is standard-library only and works offline. `README.md` is not
copied by default because the website owns its documentation hub; use
`--include-readme` to publish it as `index.html`. If the optional `g4f.dev`
checkout is not present, clone it first:

```bash
git clone --depth 1 https://github.com/gpt4free/g4f.dev.git ../g4f.dev
```

## Guides

- [Sandbox desktop](sandbox-desktop.md) — start script, clipboard, scaling,
  app volume, user id, `G4F_API_KEY`, domain routing, tray and Polli install
- [Qwen Chat sessions and automatic token refresh](qwen-sessions.md)
- [API Route authenticated provider](api-route.md)
- [Custom model routing with `config.yaml`](config-yaml-routing.md)
- [Automatic development pipeline](dev-pipeline.md)
- [Reasoning standardization](reasoning-standardization.md)
- [Build workflow](build-workflow.md)
- [ARM64 build plan](arm64-build-plan.md)
- [AArch64 compatibility](aarch64-compatibility.md)
