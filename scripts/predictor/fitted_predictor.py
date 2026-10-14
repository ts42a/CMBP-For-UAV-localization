"""Reusable source-only fits for online map queries and matched H1 comparisons."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from evaluate_predictors import TEX_COLS, EXTRA_COLS, SETTINGS, logistic_options, fit_predict, fit_fe_platt

class FittedPredictor:
    def __init__(self,kind,data,train,held_out_group):
        if kind not in ('M1_texture','M2_dino'):raise ValueError('unsupported online predictor')
        train=np.asarray(train,dtype=int)
        if not len(train) or np.any(data.group[train]==held_out_group):raise ValueError('empty or leaking fit')
        self.kind=kind;self.transforms=[];self.constant=None;self.train_groups=sorted(set(data.group[train]))
        y=data.y[train]
        if len(np.unique(y))<2:
            self.constant=float(y.mean());return
        if kind=='M1_texture':x=data.tex[train]
        else:
            parts=[]
            for a in (data.near,data.win):
                sc=StandardScaler().fit(a[train])
                pc=PCA(n_components=SETTINGS['pca_components'],random_state=SETTINGS['pca_seed'],svd_solver=SETTINGS['pca_solver']).fit(sc.transform(a[train]))
                self.transforms.append((sc,pc));parts.append(pc.transform(sc.transform(a[train])))
            x=np.hstack([*parts,data.extra[train]])
        self.scaler=StandardScaler().fit(x)
        self.model=LogisticRegression(**logistic_options(SETTINGS['model_c'],SETTINGS['model_iterations'])).fit(self.scaler.transform(x),y)

    def arrays(self,texture,near,window,extra):
        if self.constant is not None:return np.full(len(extra),self.constant)
        if self.kind=='M1_texture':x=texture
        else:
            parts=[pc.transform(sc.transform(a)) for a,(sc,pc) in zip((near,window),self.transforms)]
            x=np.hstack([*parts,extra])
        return self.model.predict_proba(self.scaler.transform(x))[:,1]

    def data(self,d,indices):
        return self.arrays(d.tex[indices],d.near[indices],d.win[indices],d.extra[indices])

    def query(self,features,near,window):
        return float(self.arrays(np.array([[features[k] for k in TEX_COLS]]),np.asarray(near,dtype=float)[None],
            np.asarray(window,dtype=float)[None],np.array([[features[k] for k in EXTRA_COLS]]))[0])

def fit_fold(data,fold):
    held=fold['fold_id'];source=np.flatnonzero(data.group!=held)
    fitted={m:FittedPredictor(m,data,source,held) for m in ('M1_texture','M2_dino')}
    oof={m:np.full(len(data.y),np.nan) for m in fitted}
    for inner in fold['inner_folds']:
        tr=np.flatnonzero(np.isin(data.group,inner['train_groups']));va=np.flatnonzero(data.group==inner['val_group'])
        if np.any(data.group[tr]==held) or np.any(data.group[va]==held):raise ValueError('outer-group leakage')
        for m in fitted:oof[m][va]=fit_predict(m,data,tr,va)
    if any(not np.isfinite(p[source]).all() for p in oof.values()):raise ValueError('incomplete source OOF')
    calibrate,slope,intercept=fit_fe_platt(oof['M2_dino'][source],data.y[source],data.group[source])
    return fitted,calibrate,{'held_out_group':held,'source_groups':sorted(set(data.group[source])),
        'fe_slope':slope,'fe_intercept':intercept}, {
        'texture':oof['M1_texture'][source],'raw':oof['M2_dino'][source],'cal':calibrate(oof['M2_dino'][source])}
