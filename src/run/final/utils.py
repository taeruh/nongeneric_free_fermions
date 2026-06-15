from matplotlib import pyplot as plt


def paper_setup():
    plt.style.use(
        [
            "./src/styles/ownstandard.mplstyle",
            "./src/styles/ownlatex.mplstyle",
            # "./src/styles/owndark.mplstyle",
        ]
    )
    plt.rcParams.update(
        {
            "figure.figsize": [*set_size()],
            "font.size": 9,
            "lines.linewidth": 1.5,
        }
    )


# get default with \the\textwidth or \the\columnwidth
# quantum;10pt;two-col \the\textwidth: width_in_pt=483.0
# revtex;10pt;two-col \the\textwidth: width_in_pt=510.0 
# revtex;10pt;two-col \the\columnwidth: width_in_pt=246.0 
def set_size(width_in_pt=246.0, height_in_width=1.0, scale=1.0):  # revtex 10pt two-col
    width_in_in = width_in_pt * scale / 72.27
    return (width_in_in, width_in_in * height_in_width)
