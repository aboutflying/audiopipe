from __future__ import annotations
import argparse
from pathlib import Path

from .runner import render_one, process_inbox

DEFAULT_WORK = Path("work")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="audiopipe")
    p.add_argument("-c", "--config", type=Path, default=None,
                   help="pipeline config; omit to use built-in defaults, "
                        "or pass a preset from config/presets/")
    p.add_argument("-w", "--work", type=Path, default=DEFAULT_WORK)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the chain on one file")
    pr.add_argument("file", type=Path)
    pr.add_argument("-o", "--out", type=Path, default=None)

    sub.add_parser("process", help="drain inbox/")

    ps = sub.add_parser("score", help="render a long-form score (many voices)")
    ps.add_argument("config", type=Path)
    ps.add_argument("-o", "--out", type=Path, required=True)

    args = p.parse_args(argv)

    if args.cmd == "run":
        out = args.out or (args.work / "outbox" / (args.file.stem + ".wav"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out_path, _, _ = render_one(args.file, args.config, out, args.work / "scratch")
        print(out_path)
    elif args.cmd == "process":
        outs = process_inbox(args.work, args.config)
        for o in outs:
            print(o)
    elif args.cmd == "score":
        from .score import render_score
        args.out.parent.mkdir(parents=True, exist_ok=True)
        placements = render_score(args.config, args.out)
        print(f"{args.out}  ({len(placements)} placements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
