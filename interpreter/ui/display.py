"""Live two-column transcript.

Subscribes to the event bus and renders both sides of the conversation as it
happens, with per-stage latency so you can see which stage to tune.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import AppConfig
from ..events import EventBus, PipelineEvent, Stage

_STAGE_ICON = {
    Stage.CAPTURED: "..",
    Stage.TRANSCRIBED: "**",
    Stage.TRANSLATED: "->",
    Stage.SPEAKING: ">>",
    Stage.DONE: "ok",
    Stage.DROPPED: "--",
    Stage.ERROR: "!!",
}


@dataclass
class _Turn:
    key: tuple[str, int]
    source_lang: str | None = None
    target_lang: str | None = None
    source_text: str = ""
    target_text: str = ""
    stage: Stage = Stage.CAPTURED
    detail: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


class LiveTranscript:
    """Renders turns into two language columns."""

    def __init__(self, cfg: AppConfig, bus: EventBus, *, console: Console | None = None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        self._turns: dict[tuple[str, int], _Turn] = {}
        self._order: deque[tuple[str, int]] = deque(maxlen=40)
        self._status = "listening"
        self._live: Live | None = None
        self._unsubscribe = None

    def __enter__(self) -> LiveTranscript:
        self._unsubscribe = self.bus.subscribe(self._on_event)
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc_info)

    def _on_event(self, event: PipelineEvent) -> None:
        key = (event.channel_id, event.seq)
        turn = self._turns.get(key)
        if turn is None:
            turn = _Turn(key=key)
            self._turns[key] = turn
            self._order.append(key)
            # Bound memory on a long conversation.
            while len(self._turns) > self._order.maxlen:
                stale = next(iter(self._turns))
                if stale in self._order:
                    break
                del self._turns[stale]

        turn.stage = event.stage
        turn.detail = event.detail or turn.detail
        turn.source_lang = event.source_lang or turn.source_lang
        turn.target_lang = event.target_lang or turn.target_lang
        turn.source_text = event.source_text or turn.source_text
        turn.target_text = event.target_text or turn.target_text
        if event.timings:
            turn.timings = dict(event.timings)

        if event.stage is Stage.SPEAKING:
            self._status = "speaking"
        elif event.stage in {Stage.DONE, Stage.DROPPED, Stage.ERROR}:
            self._status = "listening"
        elif event.stage is Stage.CAPTURED:
            self._status = "translating"

        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        left, right = [], []
        for key in self._order:
            turn = self._turns.get(key)
            if turn is None:
                continue
            rendered = self._render_turn(turn)
            if rendered is None:
                continue
            if turn.source_lang == self.cfg.language_a.code:
                left.append(rendered)
                right.append(Text(""))
            else:
                left.append(Text(""))
                right.append(rendered)

        visible = 12
        for row_left, row_right in zip(left[-visible:], right[-visible:]):
            table.add_row(row_left, row_right)

        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(ratio=1)
        header.add_row(
            Text(f" {self.cfg.language_a.name} ", style="bold cyan"),
            Text(f" {self.cfg.language_b.name} ", style="bold magenta"),
        )

        footer = Text(
            f"  {self._status}    Ctrl-C to stop",
            style="dim",
        )

        return Panel(
            Group(header, Text(""), table, Text(""), footer),
            title="duplex-interpreter",
            border_style="blue",
        )

    def _render_turn(self, turn: _Turn) -> Text | None:
        icon = _STAGE_ICON.get(turn.stage, "  ")

        if turn.stage is Stage.DROPPED:
            return Text(f"{icon} ({turn.detail})", style="dim italic")
        if turn.stage is Stage.ERROR:
            return Text(f"{icon} {turn.detail}", style="red")
        if not turn.source_text:
            return None

        text = Text()
        text.append(f"{icon} ", style="dim")
        text.append(turn.source_text, style="white")
        if turn.target_text:
            text.append("\n   ")
            style = "bold cyan" if turn.target_lang == self.cfg.language_a.code else "bold magenta"
            text.append(turn.target_text, style=style)
        if turn.stage in {Stage.SPEAKING, Stage.DONE} and turn.timings:
            parts = " ".join(f"{k} {v:.0f}ms" for k, v in turn.timings.items())
            text.append(f"\n   {parts}  |  total {turn.total_ms:.0f}ms", style="dim")
        if turn.stage is Stage.DONE and turn.detail == "interrupted":
            text.append("  (interrupted)", style="yellow")
        return text


class PlainLogger:
    """Line-per-event output for `--no-live`, pipes and CI."""

    def __init__(self, cfg: AppConfig, bus: EventBus, *, console: Console | None = None) -> None:
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        self._unsubscribe = None

    def __enter__(self) -> PlainLogger:
        self._unsubscribe = self.bus.subscribe(self._on_event)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()

    def _on_event(self, event: PipelineEvent) -> None:
        tag = f"[{event.channel_id}{event.seq}]"
        if event.stage is Stage.TRANSCRIBED:
            self.console.print(
                f"{tag} heard ({event.source_lang}): {event.source_text}", highlight=False
            )
        elif event.stage is Stage.SPEAKING:
            self.console.print(
                f"{tag} says  ({event.target_lang}): {event.target_text}"
                f"   [{event.total_ms:.0f}ms]",
                highlight=False,
            )
        elif event.stage is Stage.DROPPED:
            self.console.print(f"{tag} dropped: {event.detail}", style="dim", highlight=False)
        elif event.stage is Stage.ERROR:
            self.console.print(f"{tag} error: {event.detail}", style="red", highlight=False)
