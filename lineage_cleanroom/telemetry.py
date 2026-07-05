"""cleanroom/telemetry.py — observabilidade por sink INJETADO (baixo acoplamento).

O core (`pipeline.py`) não sabe COMO os eventos são registrados; ele só chama `emit(event, data)`.
Aqui ficam implementações de sink plugáveis (console, JSONL, fan-out). Espelha o padrão de
`libs/ai/squad.py::make_jsonl_sink`. Fail-soft: telemetria nunca derruba o pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

Emit = Callable[[str, dict], None]


def make_console_sink(prefix: str = "[cleanroom]") -> Emit:
    def emit(event: str, data: dict) -> None:
        try:
            print(f"{prefix} {event}: {json.dumps(data, sort_keys=True)}")
        except Exception:
            pass
    return emit


def make_jsonl_sink(path: str | Path) -> Emit:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def emit(event: str, data: dict) -> None:
        try:
            rec = {"ts": time.time(), "event": event, "data": data}
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception:
            pass
    return emit


def fanout(*sinks: Emit) -> Emit:
    def emit(event: str, data: dict) -> None:
        for s in sinks:
            s(event, data)
    return emit


def null_sink() -> Emit:
    def emit(event: str, data: dict) -> None:
        return None
    return emit
