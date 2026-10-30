import sys, importlib.util, numpy as np
from pathlib import Path
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark')
spec=importlib.util.spec_from_file_location('predict', B/'predict.py'); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
OUT=Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
for prob in ['diffusion_reaction','burgers','advection','poisson']:
    tr=np.load(B/'problems'/prob/'training_data.npz'); te=np.load(B/'problems'/prob/'test_data.npz')
    m=P.build_model(prob, B/'problems'/prob/'checkpoint'/'model')
    ptr=P.predict_values(m, tr['X_train0'], tr['X_train1'], 50)
    pte=P.predict_values(m, te['branch'], te['trunk'], 50)
    np.savez(OUT/f'{prob}.npz', train=ptr, test=pte)
    def rel(y,p): return np.linalg.norm(y-p,axis=1)/np.linalg.norm(y,axis=1)
    print(prob, 'train relL2 median %.4f mean %.4f max %.4f'%(np.median(rel(tr['y_train'],ptr)),rel(tr['y_train'],ptr).mean(),rel(tr['y_train'],ptr).max()),
          '| test median %.4f mean %.4f max %.4f'%(np.median(rel(te['solution'],pte)),rel(te['solution'],pte).mean(),rel(te['solution'],pte).max()), flush=True)
tr=np.load(B/'problems'/'cavity_flow'/'training_data.npz'); te=np.load(B/'problems'/'cavity_flow'/'test_data.npz')
n=tr['branch_train'].shape[0]//10201
btr=tr['branch_train'].reshape(n,10201,101)[:,0]; Ttr=tr['trunk_train'].reshape(n,10201,2)
print('cavity train samples', n, 'branch const within sample:', np.allclose(tr['branch_train'].reshape(n,10201,101), btr[:,None]))
res={}
for comp,ck in [('u','checkpoint_u'),('v','checkpoint_v')]:
    m=P.build_model('cavity_flow', B/'problems'/'cavity_flow'/ck/'model')
    res[comp+'_train']=P.predict_values(m, btr, Ttr, 5)
    res[comp+'_test']=P.predict_values(m, te['branch'], te['trunk'], 5)
np.savez(OUT/'cavity_flow.npz', **res)
print('done')
