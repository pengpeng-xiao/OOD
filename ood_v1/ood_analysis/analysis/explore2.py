import numpy as np
R='/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems/'
np.set_printoptions(precision=5, suppress=True, linewidth=160)
tr=np.load(R+'advection/training_data.npz'); te=np.load(R+'advection/test_data.npz')
print('adv train per-sample min pct', np.percentile(tr['X_train0'].min(1),[0,50,100]))
print('adv test per-sample min pct', np.percentile(te['branch'].min(1),[0,50,100]))
print('adv test per-sample max pct', np.percentile(te['branch'].max(1),[0,5,25,50,75,95,100]))
print('adv train per-sample max pct', np.percentile(tr['X_train0'].max(1),[0,5,25,50,75,95,100]))
for p in ['burgers']:
    tr=np.load(R+p+'/training_data.npz'); te=np.load(R+p+'/test_data.npz')
    X=tr['X_train0']; print('burgers train |x0-x100| pct', np.percentile(np.abs(X[:,0]-X[:,-1]),[50,90,97,99,100]))
    X=te['branch']; print('burgers test |x0-x100| pct', np.percentile(np.abs(X[:,0]-X[:,-1]),[50,90,97,99,100]))
    # IC in solution?
    y=tr['y_train']; T=tr['X_train1']
    m=T[:,1]==0; print('t=0 points',m.sum(), 'x of t=0', T[m,0][:5])
    print('burgers sol at t=0 == branch?', np.abs(y[:,m]-tr['X_train0']).max())
for p in ['diffusion_reaction','advection','poisson','burgers']:
    T=np.load(R+p+'/training_data.npz')['X_train1']
    print(p,'trunk first rows',T[:3].tolist(),'last',T[-1].tolist())
T=np.load(R+'poisson/training_data.npz')['X_train1']; y=np.load(R+'poisson/training_data.npz')['y_train']; X=np.load(R+'poisson/training_data.npz')['X_train0']
print('poisson trunk bbox', T.min(0), T.max(0))
# where does branch appear in solution?
idx=[np.argmin(np.abs(y[0]-X[0,j])) for j in range(0,101,10)]
print('poisson branch-matching trunk pts', T[idx].tolist())
