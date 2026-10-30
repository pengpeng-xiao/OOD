import numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems'); PR=Path('/mnt/ssd/pengpengxiao/ood/ood_analysis/preds')
np.set_printoptions(precision=4, suppress=True, linewidth=180)
def rel(y,p): return np.linalg.norm(y-p,axis=1)/np.linalg.norm(y,axis=1)
q=[0,5,25,50,75,90,95,99,100]
for prob in ['diffusion_reaction','burgers','advection','poisson']:
    tr=np.load(B/prob/'training_data.npz'); te=np.load(B/prob/'test_data.npz'); p=np.load(PR/f'{prob}.npz')
    etr=rel(tr['y_train'],p['train']); ete=rel(te['solution'],p['test'])
    print(prob); print('  train', np.percentile(etr,q)); print('  test ', np.percentile(ete,q))
    print('  test frac > train max:', np.mean(ete>etr.max()), ' > train p99:', np.mean(ete>np.percentile(etr,99)))
tr=np.load(B/'cavity_flow'/'training_data.npz'); te=np.load(B/'cavity_flow'/'test_data.npz'); p=np.load(PR/'cavity_flow.npz')
utr=tr['u_train'].reshape(21,-1); vtr=tr['v_train'].reshape(21,-1)
for c,ytr,yte in [('u',utr,te['velocity_u']),('v',vtr,te['velocity_v'])]:
    etr=rel(ytr,p[c+'_train']); ete=rel(yte,p[c+'_test'])
    print('cavity',c); print('  train', np.sort(etr)); print('  test ', np.sort(ete))
