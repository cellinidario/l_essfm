"""Uniform notebook figure style; all measured points are drawn without smoothing."""
from pathlib import Path
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
                 ('essfm','darkorange','s',r'ESSFM, $\rho=0.5$'),
                 ('essfmopt','m','x',r'ESSFM, opt. $\rho$' if fixed else 'ESSFM'),
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
        x=[c for c,_ in ldbp] if versus=='complexity' else [g['LDBP_TOTAL_STEPS']]*len(ldbp)
        ax.plot(x,[s for _,s in ldbp],color='g',ls='--' if versus=='complexity' else 'none',marker='*',ms=12)
        handles.append(Line2D([0],[0],color='g',marker='*',label='LDBP'))
    ax.axhline(g['ideal125'],color='r');ax.axhline(g['ideal2'],color='r',ls='--')
    ax.axhline(g['edc'],color='y')
    if versus=='complexity':
        x0=eval_compl(0,1.125,g['L_KM'],g['R_GBD'])
        ax.plot([x0],[g['edc']],color='y',marker='D')
    ax.set_xscale('log')
    # No copied limits: even low-rate complexities and SNRs must remain visible.
    ax.margins(x=.08,y=.08)
    handles.insert(0,Line2D([0],[0],color='r',label='Ideal DBP'))
    handles.append(Line2D([0],[0],color='y',label='EDC'))
    legend=ax.legend(handles=handles,loc='lower right',fontsize=11)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([0],[0],color='k',ls='-',label='$n=1.125$'),
                       Line2D([0],[0],color='k',ls='--',label='$n=2$')],loc='upper left',fontsize=13)
    ax.set_xlabel('Complexity $C_M$ [RM/2D]' if versus=='complexity' else 'Number of steps $N_s$')
    ax.set_ylabel('Effective SNR [dB]');ax.grid(True,alpha=.3)
    fig.tight_layout()
    stem='fig1b' if versus=='complexity' else 'fig_vs_steps'
    for extension in ('png','svg','pdf'):
        fig.savefig(Path(g['RESDIR'])/f'{stem}.{extension}',dpi=110)
    return fig
