"""Score the OOD detection results against the private split key (AUROC, accuracy, confusion)."""
import json, re, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
KEY=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/ssd/pengpengxiao/deeponet-extrapolation/distribution/private/legacy_with_labels_20261013/anonymous_pde_benchmark_test_key.json')  # matches this bundle (manifest 0605dffb); the top-level key is for the re-shuffled release
A=Path(__file__).resolve().parent.parent; R=A/'results'
B=Path('/mnt/ssd/pengpengxiao/ood/anonymous_pde_benchmark/problems')
key=json.load(open(KEY))['problems']
rows=[]; allY=[]; allS=[]; allP=[]
for prob in ['diffusion_reaction','burgers','advection','poisson','cavity_flow']:
    sid=np.load(B/prob/'test_data.npz')['sample_id']
    kmap={e['sample_id']:e for e in key[prob]}
    y=np.array([kmap[s]['split']=='ood' for s in sid])          # 1 = OOD
    src=np.array([kmap[s]['source'].split('/')[-1] for s in sid])
    L=np.load(R/f'{prob}_labels.npz'); pred=L['ood']
    if prob=='cavity_flow':
        scores={'lid amplitude U (primary)':L['U'],'model error (u)':L['eu']}
    else:
        scores={'GRF 2logLR (primary)':L['lr_te'],'model error':L['e_te'],'|l_hat - l_train|':np.abs(L['l_hat']-np.median(L['l_tr']))}
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    print(f'===== {prob}  (n={len(y)}, OOD={y.sum()}, ID={(~y).sum()})')
    print(f'  hard labels: accuracy={np.mean(pred==y):.4f}  TP={tp} FP={fp} TN={tn} FN={fn}')
    for name,s in scores.items():
        print(f'  AUROC {name:28s} = {roc_auc_score(y,s):.4f}')
    if prob=='advection':   # per OOD sub-regime vs ID
        for f in sorted(set(src[y]), key=lambda t: float(re.search(r'ls_([\d.]+)',t).group(1))):
            m=(src==f)|(~y)
            print(f'    {f}: AUROC(2logLR)={roc_auc_score(y[m],L["lr_te"][m]):.4f}  AUROC(err)={roc_auc_score(y[m],L["e_te"][m]):.4f}  recall={pred[src==f].mean():.3f}')
    s_primary=list(scores.values())[0]
    rows.append((prob,len(y),np.mean(pred==y),roc_auc_score(y,s_primary)))
    allY.append(y); allP.append(pred)
Y=np.concatenate(allY); P=np.concatenate(allP)
print('===== overall hard-label accuracy %.4f over %d samples; misclassified=%d'%(np.mean(P==Y),len(Y),(P!=Y).sum()))
