# Telegram sidechannel

The Chreatures build can receive guidance from the existing `@ltshitcoims_bot`
while an autonomous cycle is running. The bot remains the sole Telegram update
consumer. Its existing dispatcher accepts `/chreatures …` and `/loom …` only in
the configured operator's private chat, then writes an identity-free message to
`hbox:/tank/chreatures/telegram/inbox`.

This is a mailbox, not a remote shell. Telegram text is queued as user guidance;
the agent polls it between work cycles and decides how it applies to the active
task. No message is executed directly.

## Agent commands

Run these from the repository root:

```bash
uv run python scripts/telegram_sidechannel.py poll
uv run python scripts/telegram_sidechannel.py poll --json
uv run python scripts/telegram_sidechannel.py reply "Short progress update"
uv run python scripts/telegram_sidechannel.py reply --reply-to 123456 "Handled."
uv run python scripts/telegram_sidechannel.py ack 123456 123457
uv run python scripts/telegram_sidechannel.py status
```

`poll` is non-destructive. A message remains pending until its Telegram update ID
is passed to `ack`; acknowledgement creates a separate marker and leaves the
original inbox record intact. `reply` atomically writes a remote queue item. The
running bot picks that item up and inserts it into its existing SQLite outbox, so
delivery keeps the bot's retry and de-duplication behavior.

The queue records contain message text, update/message IDs, command, and
timestamps. They do not contain the Telegram token or the operator chat ID. All
queue directories are mode `0700` and records are mode `0600` on hbox. The bot
token stays in its pre-existing private config file and the local client never
reads it.

The bridge is polling-based rather than a live continuation channel. The agent
must run `poll` during autonomous cycles; Telegram does not inject a turn into an
already-running agent.

## Remote integration

The deployed helper is `/tank/chreatures/telegram/sidechannel.py`; its queue root
is the containing directory. The existing gate bot imports that project-owned
module and routes the two commands after its normal freshness, private-chat, and
configured-operator checks. Outgoing replies are accepted before the gate bot's
normal outbox flush. Do not start another `getUpdates` process for this bot.
