# the live two column transcript you see while its running
#
# listens to the event bus and redraws as stuff happens. shows the timings
# too so you can see which part is slow

from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..events import Stage

# little text icons instead of emoji, some terminals mangle emoji
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
    key: tuple
    source_lang: str = None
    target_lang: str = None
    source_text: str = ""
    target_text: str = ""
    stage: Stage = Stage.CAPTURED
    detail: str = None
    timings: dict = field(default_factory=dict)

    @property
    def total_ms(self):
        return sum(self.timings.values())


class LiveTranscript:
    def __init__(self, cfg, bus, *, console=None):
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        self._turns = {}
        self._order = deque(maxlen=40)
        self._status = "listening"
        self._live = None
        self._unsubscribe = None

    # these two make "with LiveTranscript(...)" work
    def __enter__(self):
        self._unsubscribe = self.bus.subscribe(self._on_event)
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info):
        if self._unsubscribe is not None:
            self._unsubscribe()
        if self._live is not None:
            self._live.update(self._render())  # one last draw
            self._live.__exit__(*exc_info)

    def _on_event(self, event):
        key = (event.channel_id, event.seq)
        turn = self._turns.get(key)

        if turn is None:
            turn = _Turn(key=key)
            self._turns[key] = turn
            self._order.append(key)
            # the deque drops the oldest key on its own, so clean up any
            # turns whose key isn't in it anymore. otherwise a long
            # conversation slowly eats memory
            while len(self._turns) > self._order.maxlen:
                stale = next(iter(self._turns))
                if stale in self._order:
                    break
                del self._turns[stale]

        # "or turn.x" so a later event that doesn't carry a field doesn't
        # wipe out what an earlier one already told us
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

        # language A goes on the left, B on the right. I add a blank to the
        # other side every time so the two lists stay the same length
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

        visible = 12  # only the last dozen fit on screen
        for row_left, row_right in zip(left[-visible:], right[-visible:]):
            table.add_row(row_left, row_right)

        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(ratio=1)
        header.add_row(
            Text(f" {self.cfg.language_a.name} ", style="bold cyan"),
            Text(f" {self.cfg.language_b.name} ", style="bold magenta"),
        )

        footer = Text(f"  {self._status}    Ctrl-C to stop", style="dim")

        return Panel(
            Group(header, Text(""), table, Text(""), footer),
            title="duplex-interpreter",
            border_style="blue",
        )

    def _render_turn(self, turn):
        icon = _STAGE_ICON.get(turn.stage, "  ")

        if turn.stage is Stage.DROPPED:
            return Text(f"{icon} ({turn.detail})", style="dim italic")
        if turn.stage is Stage.ERROR:
            return Text(f"{icon} {turn.detail}", style="red")
        if not turn.source_text:
            return None  # nothing to show yet

        text = Text()
        text.append(f"{icon} ", style="dim")
        text.append(turn.source_text, style="white")

        if turn.target_text:
            text.append("\n   ")
            if turn.target_lang == self.cfg.language_a.code:
                style = "bold cyan"
            else:
                style = "bold magenta"
            text.append(turn.target_text, style=style)

        if turn.stage in {Stage.SPEAKING, Stage.DONE} and turn.timings:
            parts = " ".join(f"{k} {v:.0f}ms" for k, v in turn.timings.items())
            text.append(f"\n   {parts}  |  total {turn.total_ms:.0f}ms", style="dim")

        if turn.stage is Stage.DONE and turn.detail == "interrupted":
            text.append("  (interrupted)", style="yellow")

        return text


class PlainLogger:
    # one line per event. for --no-live, or when you pipe the output
    # somewhere, since the fancy version makes a mess in a log file

    def __init__(self, cfg, bus, *, console=None):
        self.cfg = cfg
        self.bus = bus
        self.console = console or Console()
        self._unsubscribe = None

    def __enter__(self):
        self._unsubscribe = self.bus.subscribe(self._on_event)
        return self

    def __exit__(self, *exc_info):
        if self._unsubscribe is not None:
            self._unsubscribe()

    def _on_event(self, event):
        tag = f"[{event.channel_id}{event.seq}]"

        # highlight=False or rich colors random numbers in the transcript
        if event.stage is Stage.TRANSCRIBED:
            self.console.print(
                f"{tag} heard ({event.source_lang}): {event.source_text}",
                highlight=False,
            )
        elif event.stage is Stage.SPEAKING:
            self.console.print(
                f"{tag} says  ({event.target_lang}): {event.target_text}"
                f"   [{event.total_ms:.0f}ms]",
                highlight=False,
            )
        elif event.stage is Stage.DROPPED:
            self.console.print(
                f"{tag} dropped: {event.detail}", style="dim", highlight=False
            )
        elif event.stage is Stage.ERROR:
            self.console.print(
                f"{tag} error: {event.detail}", style="red", highlight=False
            )
