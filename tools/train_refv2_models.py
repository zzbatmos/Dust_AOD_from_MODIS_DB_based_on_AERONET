"""Fixed-form refits using the unchanged v2 fitting kernel and point targets.

The caller supplies the frozen deployment or outer-fold membership. This module
does not select a form, tune a setting, infer a fold, or publish an artifact.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import refv2_training_core as core
from refv2_target import point_target

ASSETS=('classifier.ubj','positive_fraction.ubj','direct_fraction.ubj',
        'pair.json','postprocessor.npz','smear_curve.npz','metadata.json')


def finish(folder,identity):
    folder=Path(folder)
    files={str(p.relative_to(folder)):core.file_hash(p) for p in sorted(folder.rglob('*'))
           if p.is_file() and p.name!='complete.json'}
    core.write_json(folder/'complete.json',{'identity':identity,'files':files})


def existing(folder,identity):
    folder=Path(folder)
    if not folder.exists():return False
    path=folder/'complete.json'
    if not path.exists():raise ValueError(f'Incomplete existing fit is preserved: {folder}')
    check=json.loads(path.read_text())
    if check['identity']!=identity:raise ValueError('Refusing changed fit identity')
    for name,digest in check['files'].items():
        if core.file_hash(folder/name)!=digest:raise ValueError('Completed fit changed')
    return True


def inner_pair_predictions(train,features,folder):
    parts=[]
    for fold,inner,validation,purged in core.grouped_splits(train):
        core.audit(Path(folder)/f'inner_{fold}',inner,validation,'model_target',
                   parent=train,purged=purged,role='frozen_inner')
        pair=core.XGBBackend().pair(inner,features,'model_target')
        q=core.oof_record(validation,'model_target',pair,features).reset_index(drop=True)
        q['inner_fold']=fold;parts.append(q)
    out=pd.concat(parts,ignore_index=True)
    if set(out.row_id)!=set(train.row_id) or out.row_id.duplicated().any():
        raise ValueError('Inner OOF membership changed')
    return out


def fit_deployment(base_train,features,reference,branch,folder,metadata,identity):
    folder=Path(folder)
    if existing(folder,identity):return
    started=time.monotonic();train=point_target(base_train,reference)
    core.check_rows(train);folder.mkdir(parents=True,exist_ok=False)
    selected='E3c' if branch=='full8' else 'E4'
    if metadata['selected']!=selected:raise ValueError('Frozen deployment form changed')
    core.audit(folder/'membership',train,train.iloc[:0],'model_target',role='deployment')
    oof=inner_pair_predictions(train,features,folder/'inner')
    post=core.fit_postprocessor(train,oof,'model_target',folder)
    pair=core.XGBBackend().pair(train,features,'model_target');pair.save(folder)
    direct=core.XGBBackend().direct(train,features,'model_target')
    direct.save_model(folder/'direct_fraction.ubj')
    core.build_deploy_curve(post,folder)
    core.write_json(folder/'metadata.json',metadata)
    finish(folder,identity)
    print('DEPLOYMENT_DONE',reference,branch,len(train),round(time.monotonic()-started,2),flush=True)


def fit_outer(base_train,test,features,reference,branch,folder,identity):
    folder=Path(folder)
    if existing(folder,identity):return pd.read_parquet(folder/'predictions.parquet')
    train=point_target(base_train,reference)
    for key in ['row_id','site_key','granule']:
        if set(train[key])&set(test[key]):raise ValueError(f'Outer {key} overlap')
    core.check_rows(train);core.check_rows(test)
    folder.mkdir(parents=True,exist_ok=False)
    core.audit(folder/'membership',train,test,'model_target' if 'model_target' in test else reference+'_f',role='outer')
    pair=core.XGBBackend().pair(train,features,'model_target');pair.save(folder/'pair')
    p,z,f=pair.predict(test[features].to_numpy(float))
    out=test[['row_id','site_key','granule','time','DB_AOD550','AER_AOD550']].copy()
    out['branch']=branch;out['P_raw']=p;out['positive_fraction']=f
    for name in ['Pc','fplus','fminus']:out[name]=np.nan
    if branch=='noSSA5':
        direct=core.XGBBackend().direct(train,features,'model_target')
        direct.save_model(folder/'direct_fraction.ubj')
        out['EV_fraction']=np.clip(direct.predict(test[features].to_numpy(float)),0,1)
    elif branch=='full8':
        oof=inner_pair_predictions(train,features,folder/'inner')
        iso=core.IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(
            oof.probability,oof.truth_fraction.ge(.2).astype(int))
        positive=oof.truth_fraction.ge(.2)
        residuals=oof.loc[positive,'residual_logit'].to_numpy(float)
        negative=train.model_target.lt(.2)
        fn=float(np.average(train.loc[negative,'model_target'],weights=np.ones(int(negative.sum())))) if negative.any() else 0.
        post=core.Postprocessor(iso.X_thresholds_,iso.y_thresholds_,
            residuals if len(residuals) else np.array([0.]),fn,not len(residuals))
        post.save(folder);oof.to_parquet(folder/'inner_oof.parquet',index=False)
        q=post.predict(p,z)
        out['EV_fraction']=q['E3c'];out['Pc']=q['probability_calibrated']
        out['fplus']=q['positive_fraction_star'];out['fminus']=fn
    else:raise ValueError('Unknown frozen branch')
    out['daod550_qa2']=np.where(p>=.6,f,0)*out.DB_AOD550
    out['daod550_ev']=out.EV_fraction*out.DB_AOD550
    if not np.isfinite(out.EV_fraction).all() or not out.EV_fraction.between(0,1).all():
        raise ValueError('Missing or unphysical finite-support prediction')
    out.to_parquet(folder/'predictions.parquet',index=False,compression='zstd')
    finish(folder,identity)
    return out


def write_manifest(root):
    root=Path(root);lines=[]
    for branch in ['full8','noSSA5']:
        for name in ASSETS:
            p=root/branch/name
            if not p.is_file():raise ValueError(f'Missing inference asset: {p}')
            lines.append((core.file_hash(p),f'{branch}/{name}'))
    path=root/'MANIFEST.sha256'
    with path.open('x') as stream:
        stream.write(''.join(f'{digest}  {name}\n' for digest,name in sorted(lines)))
    return core.file_hash(path)
