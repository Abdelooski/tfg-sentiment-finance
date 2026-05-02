"""
src/data/convert_wsb_sql_to_csv.py

Converts data/raw/p4p_reddit_posts.sql (MySQL dump, ~558 MB) to a flat CSV
without loading the full file into memory.

Only INSERT INTO `reddit_posts` VALUES lines are parsed. Each line may
contain thousands of row tuples separated by commas.

Columns extracted (0-based indices in the SQL tuple):
  0  -> id
  7  -> body          (selftext)
  10 -> title
  12 -> timestamp     (created_utc, already datetime, NOT unix)
  14 -> score
  15 -> symbol
  17 -> sentiment_title_score
  18 -> sentiment_body_score

Output: data/raw/reddit_wsb_large.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SQL_PATH  = ROOT / "data" / "raw" / "p4p_reddit_posts.sql"
CSV_PATH  = ROOT / "data" / "raw" / "reddit_wsb_large.csv"

# ── Schema ────────────────────────────────────────────────────────────────────
# Total number of columns in the reddit_posts table (confirmed from CREATE TABLE)
_TOTAL_COLS = 19

# Indices to keep and their output names (order matters for OUTPUT_COLS)
_WANT_IDX = [0, 10, 7, 12, 14, 15, 17, 18]
OUTPUT_COLS = [
    "id",
    "title",
    "body",
    "timestamp",
    "score",
    "symbol",
    "sentiment_title_score",
    "sentiment_body_score",
]

# Progress interval
_PROGRESS_EVERY = 10_000

# MySQL escape sequences inside single-quoted strings
_MYSQL_ESCAPES: dict[str, str] = {
    "'":  "'",
    "\\": "\\",
    "n":  "\n",
    "r":  "\r",
    "t":  "\t",
    "0":  "\x00",
    "b":  "\x08",
    "Z":  "\x1a",
}


# ── State-machine parser ──────────────────────────────────────────────────────

def _parse_values_line(line: str) -> list[list[str]]:
    """
    Parse a single MySQL INSERT VALUES line into a list of row-tuples.

    The line looks like:
        INSERT INTO `reddit_posts` VALUES (v1,v2,...),(v1,v2,...), ...;

    Returns a list of lists, each inner list containing _TOTAL_COLS string
    values. NULL becomes the empty string "".

    Raises ValueError if the line cannot be parsed.
    """
    # Strip the prefix up to and including the first '('
    prefix = "INSERT INTO `reddit_posts` VALUES "
    if not line.startswith(prefix):
        raise ValueError("line does not start with expected prefix")

    body = line[len(prefix):]
    if body.endswith(";"):
        body = body[:-1]

    rows: list[list[str]] = []
    fields: list[str] = []
    buf: list[str] = []

    # State flags
    in_string   = False   # inside single-quoted value
    expect_open = True    # expecting '(' to start a new row tuple

    i = 0
    n = len(body)

    while i < n:
        ch = body[i]

        # ── Between tuples: skip commas and whitespace, expect '(' ────────────
        if expect_open:
            if ch in (" ", "\t", ","):
                i += 1
                continue
            if ch == "(":
                fields = []
                buf    = []
                in_string   = False
                expect_open = False
                i += 1
                continue
            raise ValueError(f"expected '(' at position {i}, got {ch!r}")

        # ── Inside a tuple ────────────────────────────────────────────────────
        if in_string:
            if ch == "\\":
                # MySQL escape sequence
                if i + 1 < n:
                    nxt = body[i + 1]
                    buf.append(_MYSQL_ESCAPES.get(nxt, nxt))
                    i += 2
                else:
                    buf.append("\\")
                    i += 1
            elif ch == "'":
                # Could be end-of-string OR escaped quote ''
                if i + 1 < n and body[i + 1] == "'":
                    buf.append("'")
                    i += 2
                else:
                    # End of quoted string
                    in_string = False
                    i += 1
            else:
                buf.append(ch)
                i += 1

        else:
            # Not in a string
            if ch == "'":
                in_string = True
                i += 1

            elif ch == ",":
                # Field separator
                fields.append("".join(buf))
                buf = []
                i += 1

            elif ch == ")":
                # End of current tuple
                fields.append("".join(buf))
                buf = []

                if len(fields) != _TOTAL_COLS:
                    raise ValueError(
                        f"expected {_TOTAL_COLS} columns, got {len(fields)}"
                    )
                rows.append(fields)

                # Advance past ')' then look for ',' or end
                i += 1
                while i < n and body[i] in (" ", "\t"):
                    i += 1

                if i < n and body[i] == ",":
                    i += 1  # skip the inter-tuple comma

                expect_open = True

            elif body[i:i+4] == "NULL":
                # Unquoted NULL → empty string
                buf.append("")
                i += 4

            else:
                # Unquoted literal (number, keyword, etc.)
                buf.append(ch)
                i += 1

    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def convert(
    sql_path: Path = SQL_PATH,
    csv_path: Path = CSV_PATH,
) -> int:
    """
    Stream-parse *sql_path* and write the selected columns to *csv_path*.

    Returns the number of rows written.
    """
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL dump not found: {sql_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_skipped = 0

    print(f"  Input   ->  {sql_path.relative_to(ROOT)}")
    print(f"  Output  ->  {csv_path.relative_to(ROOT)}")
    print(f"  Columns ->  {OUTPUT_COLS}")
    print()

    with (
        sql_path.open("r", encoding="utf-8", errors="replace") as sql_fh,
        csv_path.open("w", newline="", encoding="utf-8") as csv_fh,
    ):
        writer = csv.writer(csv_fh)
        writer.writerow(OUTPUT_COLS)

        for lineno, raw_line in enumerate(sql_fh, start=1):
            line = raw_line.rstrip("\n\r")

            if not line.startswith("INSERT INTO `reddit_posts` VALUES "):
                continue

            try:
                tuples = _parse_values_line(line)
            except ValueError as exc:
                rows_skipped += 1
                print(f"  [SKIP] line {lineno}: {exc}")
                continue

            for tup in tuples:
                try:
                    row = [tup[idx] for idx in _WANT_IDX]
                except IndexError as exc:
                    rows_skipped += 1
                    print(f"  [SKIP] line {lineno}, tuple extraction: {exc}")
                    continue

                writer.writerow(row)
                rows_written += 1

                if rows_written % _PROGRESS_EVERY == 0:
                    print(f"  ... {rows_written:,} rows written "
                          f"({rows_skipped} skipped) — line {lineno:,}")

    print()
    print(f"  Done.  Rows written : {rows_written:,}")
    print(f"         Rows skipped : {rows_skipped:,}")
    print(f"  Saved  ->  {csv_path.relative_to(ROOT)}")

    return rows_written


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    convert()
