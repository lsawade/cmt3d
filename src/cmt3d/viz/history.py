import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np


def history(cmts, costs):

    fig = plt.figure(figsize=(10, 6))

    GS = gs.GridSpec(1, 2, wspace=0.2)
    gs_mt = gs.GridSpecFromSubplotSpec(3, 2, GS[0], hspace=0.3)
    gs_hypo = gs.GridSpecFromSubplotSpec(3, 2, GS[1], wspace=0.4, hspace=0.3)

    # Color each iteration to dellineate linesearches
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(cmts), endpoint=True))

    # Parameter subplot assignment
    psp = dict(
        time_shift=[0, 0],
        half_duration=[0, 1],
        latitude=[1, 0],
        longitude=[1, 1],
        depth_in_m=[2, 0],
        cost=[2, 1],
        m_rr=[0, 0],
        m_tt=[1, 0],
        m_pp=[2, 0],
        m_rt=[0, 1],
        m_rp=[1, 1],
        m_tp=[2, 1],

    )

    for par, (row, col) in psp.items():

        if "m_" in par:
            ax = fig.add_subplot(gs_mt[row, col])
        else:
            ax = fig.add_subplot(gs_hypo[row, col])

        for (it, _lscmts), (_, _lscosts) in zip(cmts.items(), costs.items()):

            m = []
            c = []

            # Get array
            iit = np.linspace(0, 1, len(_lscmts), endpoint=True)

            #
            for _i, ((ls, _cmt), (_, _cost)) in enumerate(zip(_lscmts.items(), _lscosts.items())):

                #
                c.append(_cost)

                if par == 'cost':
                    m.append(it + iit[_i])

                else:
                    _m = getattr(_cmt, par)
                    if "m_" in par:
                        _m /= _cmt.M0
                    m.append(_m)

            # # Find min/max
            # xmin, xmax = 0
            # ymin, ymax = 0
            if par == 'cost':
                ax.plot(it+iit, c)
            else:
                ax.plot(it+iit, m)

        if "m_" in par:
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["-1", "0", "1"])
            ax.plot([0, it], [0, 0], "k--", lw=0.75)
        elif par != 'cost':
            ival = getattr(cmts[0][0], par)
            ax.plot([0, it], [ival, ival], "k--", lw=0.75)

        if row == 2:
            pass
        else:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(labelbottom=False, bottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlim([0, None])
        plt.title(f"{par}")

    plt.show()
