# Public repository audit

Audited 5 September 2026 before the initial public commit. The candidate tree
was checked with Gitleaks plus targeted searches for bot tokens, API keys,
private keys, credential-bearing URLs, Telegram identity fields, and imported
conversation text. No credential or private owner/chat identifier was found.

Before publishing, keep these local operational or imported files out of the
commit:

- `AGENTS.md`
- `ARCHITECTURE.md`
- `COGNITIVE_ORGANS.md`
- `earlierresearch_plan.md`
- `og_convo.md`
- `runs/` (already ignored)

Do not use a broad `git add .` until the five root files above are ignored or
the initial commit is staged from an explicit allowlist.

The Telegram adapter is safe to publish: it contains the public bot name,
remote host alias and project queue paths, but no token, configured owner ID,
message contents, or authentication state. Live queue files and the token
configuration remain on hbox outside this repository.

The checked-in observatory artifacts contain only the synthetic residents
Mica, Fern and Pip, model telemetry, one caregiver action (`Placed food`), and
reproducibility metadata. They contain no person, account, chat, or device
telemetry. `integrations/artifacts/observatory/observatory_report.json:50`
does retain an absolute local input path; replace it with a repository-relative
path if publishing workstation paths is undesirable. The root screenshot has
no creator, location, comment, or color-profile metadata and shows only the
synthetic habitat UI.
