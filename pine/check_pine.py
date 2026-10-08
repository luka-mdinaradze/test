#!/usr/bin/env python3
"""Check Pine Script line-continuation indentation.

Pine's rule: when a statement is split across lines, each continuation line
must be indented by a number of spaces that is NOT a multiple of 4. A multiple
of 4 reads as a new code block instead, and the parser reports
"end of line without line continuation" — pointing at the line BEFORE the real
problem, which is what makes it confusing to find by eye.

    python3 pine/check_pine.py pine/btc_regime.pine
"""
import re
import sys
from pathlib import Path

CONT_END = re.compile(r"(\?|:|,|\+|-|\*|/|=|\band\b|\bor\b|\bnot\b|\()\s*$")


def strip_code(line: str) -> str:
    """Drop a trailing // comment, ignoring // inside string literals."""
    out, in_str, quote, i = [], False, "", 0
    while i < len(line):
        c = line[i]
        if in_str:
            if c == "\\":
                out.append(c)
                i += 1
                if i < len(line):
                    out.append(line[i])
                i += 1
                continue
            if c == quote:
                in_str = False
            out.append(c)
        else:
            if c in "\"'":
                in_str, quote = True, c
                out.append(c)
            elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            else:
                out.append(c)
        i += 1
    return "".join(out)


def check(path: Path) -> int:
    lines = path.read_text().split("\n")
    problems = []
    depth = 0           # unclosed ( [ at the start of the current line
    prev_open = False   # previous code line ended mid-expression

    for n, raw in enumerate(lines, 1):
        code = strip_code(raw)
        stripped = code.strip()
        if not stripped:
            continue

        indent = len(code) - len(code.lstrip(" "))
        if "\t" in code[:indent]:
            problems.append((n, "tab used for indentation (Pine wants spaces)", raw))

        is_continuation = depth > 0 or prev_open
        if is_continuation and indent % 4 == 0:
            problems.append((
                n,
                f"continuation line indented {indent} spaces, a multiple of 4 — "
                f"use {indent + 1} or {indent - 1}",
                raw))

        depth += code.count("(") - code.count(")") + code.count("[") - code.count("]")
        depth = max(depth, 0)
        prev_open = depth > 0 or bool(CONT_END.search(stripped))

    print(f"{path}: {len(lines)} lines")
    if not problems:
        print("OK — no continuation-indent problems found")
        return 0
    for n, msg, raw in problems:
        print(f"\nline {n}: {msg}")
        print(f"  | {raw}")
    print(f"\n{len(problems)} problem(s)")
    return 1


if __name__ == "__main__":
    targets = sys.argv[1:] or ["pine/btc_regime.pine"]
    sys.exit(max(check(Path(t)) for t in targets))
