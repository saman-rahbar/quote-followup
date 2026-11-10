"""Colours and chart defaults shared by every figure, so they read as one set."""

import matplotlib as mpl

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
WON, LOST = "#0ca30c", "#898781"
HIGHLIGHT = "#2a78d6"

# One colour per type of work, the same in every chart. These three are the first three of a palette
# that stays distinguishable for colour-blind readers, which matters most in a scatter.
WORKFLOW_COLOURS = {"recurring": "#2a78d6", "project": "#eb6834", "emergency": "#1baf7a"}

# The words people see, shared with the board so a chart and the list never disagree.
WORKFLOW_NAMES = {"recurring": "Recurring service", "project": "One-off project", "emergency": "Emergency job"}
CHANNEL_NAMES = {"repeat_offline": "Repeat\ncustomer", "referral": "Referred by\nsomeone",
                 "paid_search": "Paid ad", "organic": "Found you\nthemselves", "marketplace": "Marketplace"}


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
        "savefig.dpi": 160, "savefig.bbox": "tight",
        "text.parse_math": False,  # labels are full of dollar amounts
    })
