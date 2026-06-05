import numpy as np
import matplotlib.pyplot as plt

# import your methods
from methods import (
    linear_um,
    cubic_um,
    rational_um,
    os_laplacian_um,
    proposed_ace
)

# -------------------------
# Signal
# -------------------------
def create_signal():
    n = 140
    x = np.zeros(n)

    x[0:30] = 30 + 10 * np.sin(np.arange(30) * 0.8)
    x[30:65] = 220 + 10 * np.sin(np.arange(35) * 0.8)
    x[65:75] = np.linspace(220, 80, 10)
    x[75:110] = 80 + 10 * np.sin(np.arange(35) * 0.8)
    x[110:140] = 200 + 10 * np.sin(np.arange(30) * 0.8)

    return x


# -------------------------
# Plot
# -------------------------
def plot_figure_3():

    x = create_signal()

    methods = [
        ("Original", lambda x: x),
        ("Linear UM", linear_um),
        ("Cubic UM", cubic_um),
        ("Rational UM", rational_um),
        ("OS Laplacian", os_laplacian_um),
        ("Proposed ACE (YENI)", lambda x: proposed_ace(x.reshape(1, -1))[0])
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (title, func) in zip(axes, methods):
        y = func(x)

        ax.plot(y, 'k-')
        ax.set_title(title)
        ax.set_xlabel("Pixel index")
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, 270)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("figure3_final_paper.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_figure_3()