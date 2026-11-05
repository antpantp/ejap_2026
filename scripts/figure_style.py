"""Shared terminology, colours, and panel lettering for manuscript figures."""

import re

COLORS = {
    "Nested selection": "#17679c",
    "Fixed Ridge": "#9276af",
    "Fixed Elastic Net": "#c77b36",
    "Fixed PLS": "#41977e",
    "Measured": "#17679c",
    "Predicted": "#c77b36",
    "With cadence": "#17679c",
    "Without cadence": "#c77b36",
}


def procedure_label(value):
    return {"Nested family selection": "Nested selection", "Ridge": "Fixed Ridge",
            "ElasticNet": "Fixed Elastic Net", "PLS": "Fixed PLS"}.get(value, value)


def style_figure(fig):
    """Remove narrative annotations; keep participant IDs and label primary panels.

    Twin work-rate axes and colour bars do not receive separate panel letters.
    The participant-ID titles are the requested exception to title removal.
    """
    for text in list(fig.texts):
        text.set_text("")
    panels = []
    slots = set()
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        title = ax.get_title()
        ax.set_title(title if re.fullmatch(r"CT_\d+", title) else "")
        spec = ax.get_subplotspec()
        slot = str(spec) if spec is not None else tuple(ax.get_position().bounds)
        if slot in slots:
            continue
        slots.add(slot)
        panels.append(ax)
    if len(panels) > 1:
        for i, ax in enumerate(panels):
            if not any(t.get_gid() == "panel-letter" for t in ax.texts):
                text = ax.text(-.10, 1.03, chr(ord("a") + i), transform=ax.transAxes,
                               fontweight="bold", fontsize=12, ha="left", va="bottom")
                text.set_gid("panel-letter")


def save_figure(fig, *args, **kwargs):
    style_figure(fig)
    return fig.savefig(*args, **kwargs)
