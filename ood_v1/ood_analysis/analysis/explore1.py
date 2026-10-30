import numpy as np
R='/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems/'
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for p in ['diffusion_reaction','burgers','advection','poisson']:
    tr=np.load(R+p+'/training_data.npz'); te=np.load(R+p+'/test_data.npz')
    Xtr=tr['X_train0'].astype(np.float64); Xte=te['branch'].astype(np.float64)
    print('=====',p)
    print(' train mean over samples (first 5, mid, last):', Xtr.mean(0)[[0,1,2,50,99,100]])
    print(' train std per point:', Xtr.std(0)[[0,1,2,50,99,100]])
    print(' test  mean:', Xte.mean(0)[[0,1,2,50,99,100]])
    print(' test  std :', Xte.std(0)[[0,1,2,50,99,100]])
    # check endpoints equality (periodic?)
    print(' train x[0]==x[-1] frac:', np.mean(np.isclose(Xtr[:,0],Xtr[:,-1])), ' test:', np.mean(np.isclose(Xte[:,0],Xte[:,-1])))
    # empirical autocorrelation vs lag (stationary estimate)
    Z=Xtr-Xtr.mean(0)
    C=np.cov(Xtr.T)
    lags=[0,1,2,5,10,20,30,50]
    ac=[np.mean(np.diag(C,k)) for k in lags]
    print(' train cov vs lag', dict(zip(lags,np.round(ac,4))))
    Ct=np.cov(Xte.T)
    print(' test  cov vs lag', dict(zip(lags,np.round([np.mean(np.diag(Ct,k)) for k in lags],4))))
    # per sample roughness
    d1tr=np.sqrt(np.mean(np.diff(Xtr,axis=1)**2,1)); d1te=np.sqrt(np.mean(np.diff(Xte,axis=1)**2,1))
    print(' roughness train pct:', np.percentile(d1tr,[0,1,5,50,95,99,100]))
    print(' roughness test  pct:', np.percentile(d1te,[0,1,5,50,95,99,100]))
    s_tr=Xtr.std(1); s_te=Xte.std(1)
    print(' sample std train pct:', np.percentile(s_tr,[0,1,5,50,95,99,100]))
    print(' sample std test  pct:', np.percentile(s_te,[0,1,5,50,95,99,100]))
    tg=te['trunk']; print(' trunk unique x:',len(np.unique(tg[:,0])),' unique t:',len(np.unique(tg[:,1])), 'same as train trunk:', tr['X_train1'].shape==tg.shape and np.allclose(tr['X_train1'],tg))
