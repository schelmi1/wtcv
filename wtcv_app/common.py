from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
MAX_LOG_CHARS = 250_000


def trim_logs(text: str) -> str:
    if len(text) <= MAX_LOG_CHARS:
        return text
    return text[-MAX_LOG_CHARS:]


def looks_like_tqdm_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "\x1b[" in s:
        return True
    if re.search(r"\d+%\|", s) is not None:
        return True
    if ("it/s" in s and "|" in s) or ("s/it" in s and "|" in s):
        return True
    if re.search(r"\b\d+/\d+\b", s) is not None and "|" in s:
        return True
    return False


def stream_command(cmd: Sequence[str], cwd: Optional[Path] = None) -> Generator[Tuple[str, str], None, None]:
    cmd_list = [str(x) for x in cmd]
    cmd_str = shlex.join(cmd_list)
    base_logs = f"$ {cmd_str}\n\n"
    current_line = ""
    live_tqdm_line = ""
    yield cmd_str, base_logs

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
    )

    assert proc.stdout is not None

    def commit_line(line: str) -> None:
        nonlocal base_logs, live_tqdm_line
        if line == "":
            return
        if looks_like_tqdm_line(line):
            live_tqdm_line = line
        else:
            if live_tqdm_line:
                base_logs += live_tqdm_line + "\n"
                live_tqdm_line = ""
            base_logs += line + "\n"

    while True:
        chunk = proc.stdout.read(2048)
        if not chunk:
            if proc.poll() is not None:
                break
            continue

        text = chunk.decode("utf-8", errors="replace")
        for ch in text:
            if ch == "\r":
                if current_line:
                    live_tqdm_line = current_line
                    current_line = ""
            elif ch == "\n":
                commit_line(current_line)
                current_line = ""
            else:
                current_line += ch

        transient = live_tqdm_line if live_tqdm_line else current_line
        yield cmd_str, trim_logs(base_logs + transient)

    rc = proc.wait()
    commit_line(current_line)
    if live_tqdm_line:
        base_logs += live_tqdm_line + "\n"
    logs = trim_logs(base_logs + f"\n[process_exit_code={rc}]\n")
    yield cmd_str, logs


def build_bool_arg(flag_true: str, flag_false: str, value: bool) -> List[str]:
    return [flag_true if value else flag_false]


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on", "enable", "enabled", "amp"}
