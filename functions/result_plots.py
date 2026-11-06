"""Uniform notebook figure style; all measured points are drawn without smoothing."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from complexity import eval_compl, eval_compl_time


def draw(g, versus='complexity'):
    plt.close('all')
    plt.rcParams.update({'lines.linewidth':2,'font.size':16,'lines.markersize':9})
    fig,ax=plt.subplots(figsize=(8,6))
    cost=lambda ns,n,kind: eval_compl_time(ns,n,g['L_KM'],g['R_GBD'],0) if kind=='ossfm' else eval_compl(ns,n,g['L_KM'],g['R_GBD'])
    # Single-span notebooks compute only the optimised-rho ESSFM, labelled plain "ESSFM";
    # multi-span ones also show the fixed rho = 0.5 control.
    fixed='essfm125' in g
    definitions=[('lessfm','b','o','L-ESSFM'),
                 ('essfm','gray','s',r'ESSFM, $\rho=0.5$'),
                 ('essfmopt','r','x',r'ESSFM, opt. $\rho$' if fixed else 'ESSFM'),
                 ('ossfm','c','+','OSSFM')]
    handles=[]
    for kind,color,marker,label in definitions:
        if kind+'125' not in g:
            continue
        handles.append(Line2D([0],[0],color=color,marker=marker,label=label))
        for suffix,n,ls in [('125',1.125,'-'),('2',2.,'--')]:
            values=g.get(kind+suffix,{}); ns=sorted(values)
            x=[cost(k,n,kind) for k in ns] if versus=='complexity' else ns
            ax.plot(x,[values[k] for k in ns],color=color,ls=ls,marker=marker,
                    ms=6 if marker=='o' else 8,mfc='none' if n==2. else color)
    ldbp=g.get('ldbp',[])
    if ldbp:
        # ldbp_ns: the number of steps of each LDBP point (one per step count); without it,
        # all points share LDBP_TOTAL_STEPS and the steps figure has unjoined stars.
        joined=versus=='complexity' or 'ldbp_ns' in g
        x=[c for c,_ in ldbp] if versus=='complexity' else g.get('ldbp_ns',[g.get('LDBP_TOTAL_STEPS')]*len(ldbp))
        ax.plot(x,[s for _,s in ldbp],color='g',ls='--' if joined else 'none',marker='*',ms=12)
        handles.append(Line2D([0],[0],color='g',marker='*',ls='--' if joined else 'none',label='LDBP'))
    # Ideal DBP in black, so that the L-ESSFM (blue) and ESSFM (red) curves stand apart.
    ax.axhline(g['ideal125'],color='k');ax.axhline(g['ideal2'],color='k',ls='--')
    if versus=='complexity':
        x0=eval_compl(0,1.125,g['L_KM'],g['R_GBD'])
        ax.plot([x0],[g['edc']],color='y',marker='D')
    else:
        ax.axhline(g['edc'],color='y')
    ax.set_xscale('log')
    # No copied limits: even low-rate complexities and SNRs must remain visible.
    ax.margins(x=.08,y=.08)
    if versus=='complexity':
        # Start at the decade below the EDC point, so that its complexity can be read, and
        # draw the EDC reference only from that point on: EDC has a single complexity.
        ax.set_xlim(left=10**np.floor(np.log10(x0)))
        right=ax.get_xlim()[1]
        ax.hlines(g['edc'],x0,right,color='y')
        ax.set_xlim(right=right)
    handles.insert(0,Line2D([0],[0],color='k',label='Ideal DBP'))
    handles.append(Line2D([0],[0],color='y',marker='D' if versus=='complexity' else None,label='EDC'))
    # Method legend inside the axes, lower right (Dario, 2026-11-06).
    legend=ax.legend(handles=handles,loc='lower right',fontsize=12)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([0],[0],color='k',ls='-',label='$n=1.125$'),
                       Line2D([0],[0],color='k',ls='--',label='$n=2$')],loc='upper left',fontsize=13)
    ax.set_xlabel('Complexity $C_M$ [RM/2D]' if versus=='complexity' else 'Number of steps $N_s$')
    ax.set_ylabel('Effective SNR [dB]');ax.grid(True,alpha=.3)
    fig.tight_layout()
    stem='fig1b' if versus=='complexity' else 'fig_vs_steps'
    for extension in ('png','svg','pdf'):
        fig.savefig(Path(g['RESDIR'])/f'{stem}.{extension}',dpi=110,bbox_inches='tight',
                    bbox_extra_artists=[legend])
    return fig
