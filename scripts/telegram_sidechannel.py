#!/usr/bin/env python3
"""Poll and answer the owner-authenticated Chreatures Telegram sidechannel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chreatures.telegram_channel import SidechannelError, TelegramChannel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chreatures Telegram sidechannel over SSH")
    parser.add_argument("--host", default="hbox")
    commands = parser.add_subparsers(dest="command", required=True)

    poll = commands.add_parser("poll", help="show unacknowledged messages without deleting them")
    poll.add_argument("--all", action="store_true", help="include acknowledged messages")
    poll.add_argument("--limit", type=int, default=100)
    poll.add_argument("--json", action="store_true")

    ack = commands.add_parser("ack", help="mark one or more inbox update ids acknowledged")
    ack.add_argument("ids", nargs="+", type=int)

    reply = commands.add_parser("reply", help="queue a reply through the bot's durable outbox")
    reply.add_argument("text")
    reply.add_argument("--reply-to", type=int)

    commands.add_parser("status", help="show queue and bot-service health")
    return parser


def main() -> None:
    args = _parser().parse_args()
    channel = TelegramChannel(args.host)
    try:
        if args.command == "poll":
            messages = channel.poll(include_acked=args.all, limit=args.limit)
            if args.json:
                print(json.dumps([asdict(message) for message in messages], ensure_ascii=False))
            elif not messages:
                print("No unacknowledged Telegram messages.")
            else:
                for message in messages:
                    print(f"[{message.id}] {message.command} {message.text}")
        elif args.command == "ack":
            ids = channel.acknowledge(args.ids)
            print("Acknowledged: " + ", ".join(str(value) for value in ids))
        elif args.command == "reply":
            reply_id = channel.reply(args.text, reply_to=args.reply_to)
            print(f"Queued reply {reply_id}.")
        else:
            status = channel.status()
            print(json.dumps(status, indent=2, sort_keys=True))
    except SidechannelError as exc:
        raise SystemExit(f"sidechannel: {exc}") from None


if __name__ == "__main__":
    main()
