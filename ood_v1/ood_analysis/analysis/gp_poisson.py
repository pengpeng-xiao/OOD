import numpy as np, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('g','gp_mle2.py')
src=open('gp_mle2.py').read().split("cfg={")[0]; ns={}; exec(src, ns)
B=ns['B']; LS=ns['LS']; profile=ns['profile']
tr=np.load(B/'poisson/training_data.npz')['X_train0'].astype(np.float64)
te=np.load(B/'poisson/test_data.npz')['branch'].astype(np.float64)
q=[0,1,5,25,50,75,95,99,100]
bins=np.r_[0.02,0.06,0.085,0.115,0.15,0.175,0.225,0.275,0.325,0.375,0.45,0.55,0.7,0.9,1.1,1.4,2.0]
for nug in [1e-12,1e-11,1e-10,1e-9]:
    o={}
    for n,X in [('tr',tr),('te',te)]:
        LL,S2=profile(X,'rbf',nug); j=LL.argmax(1); o['l_'+n]=LS[j]; o['s_'+n]=np.sqrt(S2[np.arange(len(X)),j]); o['LL_'+n]=LL
    print('nug',nug,'| l train',np.percentile(o['l_tr'],q),'\n   l test ',np.percentile(o['l_te'],q),'\n   s train',np.round(np.percentile(o['s_tr'],[1,50,99]),3),'s test',np.round(np.percentile(o['s_te'],[1,50,99]),3))
    print('   hist train',np.histogram(o['l_tr'],bins)[0],'\n   hist test ',np.histogram(o['l_te'],bins)[0])
    if nug==1e-10: np.savez('/mnt/ssd/pengpengxiao/ood/ood_analysis/feat/poisson_gp2.npz', LS=LS, **o)
