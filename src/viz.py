"""Estilo de gráficos compartido por notebooks y dashboard."""
import matplotlib.pyplot as plt
import seaborn as sns

# Paleta categórica validada para daltonismo (orden fijo, no se cicla)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = "Blues"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Color y marcador fijos por segmento: el color sigue a la entidad, no al ranking
SEGMENT_ORDER = ["Campeones", "Alto valor en riesgo", "Leales recientes", "Nuevos", "Bajo valor inactivos"]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, PALETTE))
SEGMENT_MARKERS = dict(zip(SEGMENT_ORDER, ["o", "s", "P", "^", "D"]))


def segment_order(present):
    """Segmentos presentes, en el orden fijo de SEGMENT_ORDER."""
    return [s for s in SEGMENT_ORDER if s in set(present)] + sorted(set(present) - set(SEGMENT_ORDER))


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 110,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })
