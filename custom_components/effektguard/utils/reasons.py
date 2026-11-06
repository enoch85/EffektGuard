"""How a layer words why it voted as it did, the same in every layer.

A reason names what the vote is about, then each part that changed the layer's own offset, every
one as cause → amount: "DM -800 past -646 → +7.0, room warming 0.30°C/h → -2.8". The layer's own
amount shows only where such parts follow, so the parts add up to the vote; the vote's offset and
weight, which the reasoning shows in front of the reason, are not said again.
"""

SHORT_RUN = "short run"


def adjusted(what: str, asked: float, got: float, why: str = "") -> str:
    """What changed the offset after the layers voted, said in words, because it never adds a vote
    of its own: "Heater guard lowered +0.4 to +0.0 (DM -600, heater starts at -460)"."""
    verb = "lowered" if got < asked else "raised"
    return f"{what} {verb} {asked:+.1f} to {got:+.1f}" + (f" ({why})" if why else "")


def change(what: str, amount: float, action: str = "") -> str:
    """One part that changed a vote, with what it added: "room warming 0.30°C/h → -0.6", or with
    what it did as well: "price +71% in 2h → pre-heat +2.0"."""
    return f"{what} → {f'{action} ' if action else ''}{amount:+.1f}"


def explain(subject: str, amount: float, changes: list[str], weight_cut: bool = False) -> str:
    """A layer's reason: what the vote is about, then what changed it.

    Args:
        subject: What the vote is about, e.g. "very cheap night" or "DM -800 past -646"
        amount: The layer's own offset, before the changes
        changes: What changed it, each as `change()` words it, or a limit such as "held at -2.0"
        weight_cut: The vote's weight was cut because the cheap price run is short

    Returns:
        The reason
    """
    parts = [change(subject, amount) if changes else subject, *changes]
    if weight_cut:
        said = any(part.startswith(SHORT_RUN) for part in changes)
        parts.append("weight cut" if said else f"{SHORT_RUN} → weight cut")
    return ", ".join(parts)
