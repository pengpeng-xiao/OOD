import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/results')
ID,OOD,TR='#2a78d6','#eb6834','#a3a29d'; INK,INK2='#0b0b0b','#52514e'
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#c3c2b7','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#ecebe7','grid.linewidth':0.6,'axes.axisbelow':True})
probs=['diffusion_reaction','burgers','advection','poisson','cavity_flow']
fig,ax=plt.subplots(2,5,figsize=(17,6.6),facecolor='#fcfcfb')
rng=np.random.default_rng(0)
for c,p in enumerate(probs):
    a0,a1=ax[0,c],ax[1,c]
    L=np.load(R/f'{p}_labels.npz')
    if p!='cavity_flow':
        ood=L['ood']; lte=L['l_hat']; ltr=L['l_tr']
        bins=np.linspace(0.02,1.2,119)
        a0.hist(ltr,bins,color=TR,label=f'train (n={len(ltr)})',density=True,edgecolor='#fcfcfb',linewidth=0.5)
        a0.hist(lte[~ood],bins,color=ID,label=f'test ID (n={(~ood).sum()})',density=True,alpha=0.85,edgecolor='#fcfcfb',linewidth=0.5)
        a0.hist(lte[ood],bins,color=OOD,label=f'test OOD (n={ood.sum()})',density=True,alpha=0.85,edgecolor='#fcfcfb',linewidth=0.5)
        a0.set_xlim(0,1.15); a0.set_xlabel('fitted GRF length scale  l̂'); a0.set_yticks([])
        lr=2*np.maximum(L['lr_te'],1e-3); lrt=2*np.maximum(L['lr_tr'],1e-3)
        a1.scatter(lrt,L['e_tr'],s=8,color=TR,alpha=0.5,linewidths=0,label='train')
        a1.scatter(lr[~ood],L['e_te'][~ood],s=14,color=ID,edgecolors='#fcfcfb',linewidths=0.5,label='test ID')
        a1.scatter(lr[ood],L['e_te'][ood],s=14,color=OOD,edgecolors='#fcfcfb',linewidths=0.5,label='test OOD')
        a1.set_xscale('log'); a1.set_yscale('log'); a1.axvline(15.1,color=INK2,lw=1,ls='--')
        a1.text(15.1,a1.get_ylim()[1] if False else 1.6,' p=1e-4',color=INK2,fontsize=8,va='top')
        a1.set_xlabel('2·log LR vs training kernel'); a1.set_ylabel('DeepONet rel. L2 error' if c==0 else '')
        a1.set_ylim(1e-4 if p=='poisson' else 1e-3, 2.5)
    else:
        ood=L['ood']; U=L['U']; Utr=L['Utr']
        a0.scatter(Utr,np.zeros_like(Utr)+0.0,marker='|',s=200,color=TR,label=f'train (n={len(Utr)})')
        a0.scatter(U[~ood],np.ones((~ood).sum()),marker='|',s=200,color=ID,label=f'test ID (n={(~ood).sum()})')
        a0.scatter(U[ood],np.ones(ood.sum()),marker='|',s=200,color=OOD,label=f'test OOD (n={ood.sum()})')
        a0.axvline(0.4,color=INK2,lw=1,ls='--'); a0.set_ylim(-0.6,1.8); a0.set_yticks([0,1]); a0.set_yticklabels(['train','test'])
        a0.set_xlabel('lid amplitude U   (branch = U·(1−x))')
        a1.scatter(Utr,L['eu_tr'],s=16,color=TR,label='train')
        a1.scatter(U[~ood],L['eu'][~ood],s=18,color=ID,edgecolors='#fcfcfb',linewidths=0.5,label='test ID')
        a1.scatter(U[ood],L['eu'][ood],s=18,color=OOD,edgecolors='#fcfcfb',linewidths=0.5,label='test OOD')
        a1.axvline(0.4,color=INK2,lw=1,ls='--'); a1.set_xlabel('lid amplitude U'); a1.set_yscale('log')
    a0.set_title(p.replace('_',' '),color=INK,fontsize=11,loc='left',fontweight='bold')
    a0.legend(frameon=False,fontsize=7.5,loc='upper right' if p not in ('cavity_flow',) else 'upper left')
ax[1,0].legend(frameon=False,fontsize=7.5,loc='lower right')
fig.suptitle('Test-set membership: inputs fit by the training GRF (blue) vs. different regimes (orange)',x=0.01,ha='left',color=INK,fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(R/'ood_summary.png',dpi=150,facecolor=fig.get_facecolor())
print('saved')
