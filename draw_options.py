""" Module drawn all OptionGraph. """

import matplotlib.pyplot as plt
from gym_minigrid.options.options import ALL_OPTIONS

for name, option in ALL_OPTIONS.items():
    print(name)
    fig, ax = plt.subplots()
    fig.set_facecolor('#181a1b')
    ax.set_facecolor('#181a1b')
    option.graph.draw(ax, fontcolor='white')
    ax.set_axis_off()
    dpi = 96
    width, height = (1056, 719)
    fig.set_size_inches(width/dpi, height/dpi)
    plt.tight_layout()
    plt.title(name)
    plt.show()
