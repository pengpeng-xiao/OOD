import numpy as np, csv
from pathlib import Path
from scipy import stats
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
S=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis')
F=S/'feat'; PR=S/'preds'; OUT=S/'results'; OUT.mkdir(exist_ok=True)
def rel(y,p): return np.linalg.norm(y-p,axis=1)/np.linalg.norm(y,axis=1)
rows=[]; summary={}
for prob in ['diffusion_reaction','burgers','advection','poisson']:
    g=np.load(F/f'{prob}_gp2.npz'); LS=g['LS']
    l0=float(np.median(g['l_tr'])); j0=int(np.argmin(np.abs(LS-l0)))
    lr_tr=g['LL_tr'].max(1)-g['LL_tr'][:,j0]; lr_te=g['LL_te'].max(1)-g['LL_te'][:,j0]
    p_emp=(1+(lr_tr[None,:]>=lr_te[:,None]).sum(1))/(len(lr_tr)+1)
    p_chi=stats.chi2.sf(2*lr_te,1)
    tr=np.load(B/prob/'training_data.npz'); te=np.load(B/prob/'test_data.npz'); pr=np.load(PR/f'{prob}.npz')
    e_tr=rel(tr['y_train'],pr['train']); e_te=rel(te['solution'],pr['test'])
    ood=p_chi<1e-4      # decision rule: GRF-likelihood-ratio test vs training kernel
    ltr=g['l_tr']; lte=g['l_te']
    print(f'===== {prob}: training l0={l0:.3f}; train l_hat range [{ltr.min():.3f},{ltr.max():.3f}]; train max 2LR={2*lr_tr.max():.2f}')
    print(f'  ID  : n={(~ood).sum():3d}  l_hat [{lte[~ood].min():.3f},{lte[~ood].max():.3f}] max 2LR={2*lr_te[~ood].max():.2f}  err median {np.median(e_te[~ood]):.4f} (train median {np.median(e_tr):.4f}, max {e_tr.max():.4f})')
    print(f'  OOD : n={ood.sum():3d}  l_hat [{lte[ood].min():.3f},{lte[ood].max():.3f}] min 2LR={2*lr_te[ood].min():.1f}  err median {np.median(e_te[ood]):.4f}')
    # sub-regimes among OOD by l_hat clusters (gaps > 0.02)
    lo=np.sort(lte[ood]); cuts=np.where(np.diff(lo)>0.02)[0]
    groups=np.split(lo,cuts+1)
    for gg in groups:
        m=ood&(lte>=gg.min())&(lte<=gg.max())
        print(f'     OOD cluster l_hat~{np.median(gg):.3f} (n={len(gg)}): err median {np.median(e_te[m]):.4f}, min {e_te[m].min():.4f}, max {e_te[m].max():.4f}')
    # two-sample checks ID-test vs train: sigma_hat, l_hat, model error
    print('  KS ID-test vs train: l_hat p=%.3f  sigma_hat p=%.3f  model-err p=%.3g'%(stats.ks_2samp(lte[~ood],ltr).pvalue, stats.ks_2samp(g['s_te'][~ood],g['s_tr']).pvalue, stats.ks_2samp(e_te[~ood],e_tr).pvalue))
    summary[prob]=(int((~ood).sum()),int(ood.sum()))
    for i,sid in enumerate(te['sample_id']):
        rows.append(dict(problem=prob,sample_id=sid,label='OOD' if ood[i] else 'ID',l_hat=round(float(lte[i]),4),sigma_hat=round(float(g['s_te'][i]),4),
                         two_log_LR=round(float(2*lr_te[i]),3),p_value_chi2=float(p_chi[i]),p_value_empirical=round(float(p_emp[i]),5),model_rel_l2=round(float(e_te[i]),5)))
    np.savez(OUT/f'{prob}_labels.npz', ood=ood, l_hat=lte, l_tr=ltr, e_te=e_te, e_tr=e_tr, lr_te=lr_te, lr_tr=lr_tr)
# cavity: branch = U*(1-x); training U grid
tr=np.load(B/'cavity_flow/training_data.npz'); te=np.load(B/'cavity_flow/test_data.npz'); pr=np.load(PR/'cavity_flow.npz')
Utr=tr['branch_train'].reshape(21,10201,101)[:,0,0]; Ute=te['branch'][:,0]
x=np.linspace(0,1,101); print('cavity branch exactly U*(1-x):', np.allclose(te['branch'], Ute[:,None]*(1-x)), np.allclose(tr['branch_train'].reshape(21,10201,101)[:,0], Utr[:,None]*(1-x)))
eu=rel(te['velocity_u'],pr['u_test']); ev=rel(te['velocity_v'],pr['v_test'])
eu_tr=rel(tr['u_train'].reshape(21,-1),pr['u_train'])
ood=Ute>Utr.max()+1e-9
print(f'===== cavity_flow: train U in [{Utr.min()},{Utr.max()}] step {np.diff(np.sort(Utr)).mean():.3f}')
print(f'  ID n={(~ood).sum()} U in [{Ute[~ood].min()},{Ute[~ood].max()}]  err_u max {eu[~ood].max():.4f} (train max {eu_tr.max():.4f})')
print(f'  OOD n={ood.sum()} U in [{Ute[ood].min()},{Ute[ood].max()}]  err_u min {eu[ood].min():.4f}')
print('  corr(U, err_u) among OOD:', np.corrcoef(Ute[ood],eu[ood])[0,1])
summary['cavity_flow']=(int((~ood).sum()),int(ood.sum()))
for i,sid in enumerate(te['sample_id']):
    rows.append(dict(problem='cavity_flow',sample_id=sid,label='OOD' if ood[i] else 'ID',lid_U=float(Ute[i]),model_rel_l2_u=round(float(eu[i]),5),model_rel_l2_v=round(float(ev[i]),5)))
np.savez(OUT/'cavity_flow_labels.npz', ood=ood, U=Ute, Utr=Utr, eu=eu, ev=ev, eu_tr=eu_tr)
keys=['problem','sample_id','label','l_hat','sigma_hat','two_log_LR','p_value_chi2','p_value_empirical','model_rel_l2','lid_U','model_rel_l2_u','model_rel_l2_v']
with open(OUT/'ood_labels.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print(summary)
