"""Durable, operator-only Telegram sidechannel for long-running Chreatures work.

The remote bot remains the only Telegram ``getUpdates`` consumer.  Its existing
dispatcher calls :class:`DurableSidechannel` only after authenticating the
configured operator.  Local tooling talks to this module over SSH and never
reads or transports the bot token or the operator chat id.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_REMOTE_ROOT = Path("/tank/chreatures/telegram")
DEFAULT_REMOTE_HELPER = DEFAULT_REMOTE_ROOT / "sidechannel.py"
MAX_TEXT_LENGTH = 4000


class SidechannelError(RuntimeError):
    """A queue record or remote sidechannel operation was invalid."""


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    id: int
    text: str
    command: str
    received_at: float
    sent_at: float | None = None
    message_id: int | None = None


def _atomic_json(path: Path, value: object) -> None:
    """Create a private JSON file atomically and fsync its directory."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SidechannelError(f"invalid sidechannel record {path.name}") from exc
    if not isinstance(value, dict):
        raise SidechannelError(f"invalid sidechannel record {path.name}")
    return value


class DurableSidechannel:
    """Filesystem queue used by the bot dispatcher and the remote helper CLI."""

    def __init__(self, root: Path = DEFAULT_REMOTE_ROOT):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.acked = self.root / "acked"
        self.outbox = self.root / "outbox"
        self.accepted = self.root / "accepted"
        for directory in (self.root, self.inbox, self.acked, self.outbox, self.accepted):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(directory, 0o700)

    @staticmethod
    def _inbox_name(update_id: int) -> str:
        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 0:
            raise SidechannelError("Telegram update id must be a non-negative integer")
        return f"tg-{update_id:020d}.json"

    def capture_update(
        self,
        update_id: int,
        text: str,
        command: str,
        *,
        message_id: int | None = None,
        sent_at: float | None = None,
        received_at: float | None = None,
    ) -> TelegramMessage:
        """Persist one already-authenticated update without any Telegram identity."""

        if command not in ("/chreatures", "/loom"):
            raise SidechannelError("unsupported sidechannel command")
        clean = text.strip()
        if not clean:
            raise SidechannelError("sidechannel message is empty")
        if len(clean) > MAX_TEXT_LENGTH:
            raise SidechannelError(f"sidechannel message exceeds {MAX_TEXT_LENGTH} characters")
        received = time.time() if received_at is None else float(received_at)
        message = TelegramMessage(
            id=update_id,
            text=clean,
            command=command,
            received_at=received,
            sent_at=float(sent_at) if sent_at is not None else None,
            message_id=message_id if isinstance(message_id, int) and not isinstance(message_id, bool) else None,
        )
        path = self.inbox / self._inbox_name(update_id)
        if path.exists():
            return self._decode_message(path)
        _atomic_json(path, asdict(message))
        return message

    def _decode_message(self, path: Path) -> TelegramMessage:
        value = _load_json(path)
        try:
            message = TelegramMessage(
                id=int(value["id"]),
                text=str(value["text"]),
                command=str(value["command"]),
                received_at=float(value["received_at"]),
                sent_at=float(value["sent_at"]) if value.get("sent_at") is not None else None,
                message_id=int(value["message_id"]) if value.get("message_id") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SidechannelError(f"invalid sidechannel record {path.name}") from exc
        if path.name != self._inbox_name(message.id):
            raise SidechannelError(f"sidechannel id does not match {path.name}")
        return message

    def poll(self, *, include_acked: bool = False, limit: int = 100) -> list[TelegramMessage]:
        if limit < 1:
            raise SidechannelError("poll limit must be positive")
        messages: list[TelegramMessage] = []
        for path in sorted(self.inbox.glob("tg-*.json")):
            message = self._decode_message(path)
            marker = self.acked / self._inbox_name(message.id)
            if include_acked or not marker.exists():
                messages.append(message)
            if len(messages) >= limit:
                break
        return messages

    def acknowledge(self, ids: Iterable[int]) -> list[int]:
        acknowledged: list[int] = []
        for update_id in dict.fromkeys(ids):
            name = self._inbox_name(update_id)
            if not (self.inbox / name).is_file():
                raise SidechannelError(f"unknown inbox id {update_id}")
            marker = self.acked / name
            if not marker.exists():
                _atomic_json(marker, {"id": update_id, "acked_at": time.time()})
            acknowledged.append(update_id)
        return acknowledged

    def queue_reply(self, text: str, *, reply_to: int | None = None) -> str:
        clean = text.strip()
        if not clean:
            raise SidechannelError("reply is empty")
        if len(clean) > MAX_TEXT_LENGTH:
            raise SidechannelError(f"reply exceeds {MAX_TEXT_LENGTH} characters")
        if reply_to is not None:
            self._inbox_name(reply_to)
        item_id = f"reply-{time.time_ns()}-{uuid.uuid4().hex[:12]}"
        _atomic_json(
            self.outbox / f"{item_id}.json",
            {"id": item_id, "text": clean, "reply_to": reply_to, "created_at": time.time()},
        )
        return item_id

    def accept_outgoing(self, enqueue: Callable[[str, str], None]) -> int:
        """Move replies into the host bot's established durable Telegram outbox."""

        count = 0
        for path in sorted(self.outbox.glob("reply-*.json")):
            value = _load_json(path)
            item_id = value.get("id")
            text = value.get("text")
            if not isinstance(item_id, str) or not isinstance(text, str):
                raise SidechannelError(f"invalid sidechannel record {path.name}")
            enqueue(item_id, text)
            os.replace(path, self.accepted / path.name)
            count += 1
        return count

    def status(self) -> dict[str, object]:
        inbox = list(self.inbox.glob("tg-*.json"))
        acknowledged = {path.name for path in self.acked.glob("tg-*.json")}
        pending = [path for path in inbox if path.name not in acknowledged]
        newest = max((path.stat().st_mtime for path in inbox), default=None)
        return {
            "root": str(self.root),
            "pending_inbox": len(pending),
            "acked_inbox": len(inbox) - len(pending),
            "queued_replies": len(list(self.outbox.glob("reply-*.json"))),
            "accepted_replies": len(list(self.accepted.glob("reply-*.json"))),
            "newest_inbox_at": newest,
        }


class TelegramChannel:
    """Local SSH transport for the project-owned queue on hbox."""

    def __init__(self, host: str = "hbox", helper: Path = DEFAULT_REMOTE_HELPER):
        self.host = host
        self.helper = Path(helper)

    def _call(self, action: str, payload: dict | None = None) -> object:
        command = ["ssh", self.host, "python3", str(self.helper), action, "--json"]
        result = subprocess.run(
            command,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "remote command failed"
            raise SidechannelError(detail)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SidechannelError("remote sidechannel returned invalid JSON") from exc

    def poll(self, *, include_acked: bool = False, limit: int = 100) -> list[TelegramMessage]:
        values = self._call("poll", {"include_acked": include_acked, "limit": limit})
        if not isinstance(values, list):
            raise SidechannelError("remote poll returned an invalid value")
        return [TelegramMessage(**value) for value in values]

    def acknowledge(self, ids: Iterable[int]) -> list[int]:
        values = self._call("ack", {"ids": list(ids)})
        if not isinstance(values, list):
            raise SidechannelError("remote acknowledgement returned an invalid value")
        return [int(value) for value in values]

    def reply(self, text: str, *, reply_to: int | None = None) -> str:
        value = self._call("reply", {"text": text, "reply_to": reply_to})
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise SidechannelError("remote reply returned an invalid value")
        return value["id"]

    def status(self) -> dict[str, object]:
        value = self._call("status")
        if not isinstance(value, dict):
            raise SidechannelError("remote status returned an invalid value")
        return value


def _remote_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chreatures Telegram remote queue helper")
    parser.add_argument("action", choices=("poll", "ack", "reply", "status"))
    parser.add_argument("--json", action="store_true", help="read JSON parameters from stdin and emit JSON")
    parser.add_argument("--root", type=Path, default=DEFAULT_REMOTE_ROOT)
    return parser


def remote_main(argv: list[str] | None = None) -> int:
    args = _remote_parser().parse_args(argv)
    payload = json.load(sys.stdin) if args.json and args.action != "status" else {}
    queue = DurableSidechannel(args.root)
    if args.action == "poll":
        result: object = [
            asdict(message)
            for message in queue.poll(
                include_acked=bool(payload.get("include_acked", False)),
                limit=int(payload.get("limit", 100)),
            )
        ]
    elif args.action == "ack":
        ids = payload.get("ids", [])
        if not isinstance(ids, list):
            raise SidechannelError("ids must be a list")
        result = queue.acknowledge(int(value) for value in ids)
    elif args.action == "reply":
        reply_to = payload.get("reply_to")
        result = {
            "id": queue.queue_reply(
                str(payload.get("text", "")),
                reply_to=int(reply_to) if reply_to is not None else None,
            )
        }
    else:
        result = queue.status()
        service = subprocess.run(
            ["systemctl", "--user", "is-active", "dregg-gatebot.service"],
            text=True,
            capture_output=True,
            check=False,
        )
        result["bot_service"] = service.stdout.strip() or "unknown"
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(remote_main())
    except SidechannelError as exc:
        raise SystemExit(str(exc)) from None
