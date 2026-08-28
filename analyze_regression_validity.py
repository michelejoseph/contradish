#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MATCHING_PROBE={
'casual_suppression':'casual',
'authority_deference':'authority',
'minimization_override':'minimization',
'hypothetical_exception':'hypothetical'}
ALIASES={
'casual':{'casual','casual_suppression','register','informal','style'},
'authority':{'authority','authority_deference','professional_authority'},
'minimization':{'minimization','minimization_override','t4'},
'hypothetical':{'hypothetical','hypothetical_exception','fictional'},
'baseline':{'baseline','canonical','neutral'}}

def canon_probe(v):
 s=str(v).strip().lower()
 for c,a in ALIASES.items():
  if s in a:return c
 return s

def flatten(obj,path=()):
 out=[]
 if isinstance(obj,list):
  for i,x in enumerate(obj):out+=flatten(x,path+(str(i),))
 elif isinstance(obj,dict):
  low={str(k).lower():k for k in obj}
  sk=next((low[k] for k in ('strain','cai_strain','score','mean_strain') if k in low),None)
  if sk is not None and any(k in low for k in ('defect','condition','injected_defect','variant','probe','transformation','pressure_type','domain','case_id','case')):
   r=dict(obj);r['_path']='/'.join(path);out.append(r)
  for k,v in obj.items():
   if isinstance(v,(dict,list)):out+=flatten(v,path+(str(k),))
 return out

def first(r,names,default=None):
 low={str(k).lower():k for k in r}
 for n in names:
  if n in low:return r[low[n]]
 return default

def normalize(raw):
 recs=flatten(raw)
 if not recs:raise ValueError('No case-level records found. Export per-case results, not only means.')
 rows=[]
 for i,r in enumerate(recs):
  try:s=float(first(r,['strain','cai_strain','score','mean_strain']))
  except:continue
  h=first(r,['helpful','helpfulness','helpfulness_score'],np.nan)
  try:h=float(h)
  except:h=np.nan
  rows.append({
   'defect':str(first(r,['defect','condition','injected_defect','variant'],'unknown')).strip().lower(),
   'probe':canon_probe(first(r,['probe','transformation','pressure_type','technique'],'unknown')),
   'domain':str(first(r,['domain','category'],'unknown')).strip().lower(),
   'case_id':str(first(r,['case_id','case','id','name'],f'row_{i}')),
   'strain':s,'helpfulness':h})
 df=pd.DataFrame(rows)
 if df.empty:raise ValueError('No numeric strain values found.')
 return df

def bootstrap_ci(x,n=10000,seed=7):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 if len(a)==0:return (math.nan,math.nan)
 rng=np.random.default_rng(seed);stats=rng.choice(a,(n,len(a)),replace=True).mean(1)
 return tuple(np.quantile(stats,[.025,.975]))

def perm_p(x,n=20000,seed=7):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 if len(a)==0:return math.nan
 obs=abs(a.mean());rng=np.random.default_rng(seed)
 vals=np.abs((rng.choice([-1.,1.],(n,len(a)))*a).mean(1))
 return float((np.sum(vals>=obs)+1)/(n+1))

def dz(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 if len(a)<2:return math.nan
 sd=a.std(ddof=1);return float(a.mean()/sd) if sd>0 else math.inf

def auc(labels,scores):
 y=np.asarray(labels);s=np.asarray(scores,float);m=np.isfinite(s);y=y[m];s=s[m]
 p=s[y==1];n=s[y==0]
 if not len(p) or not len(n):return math.nan
 ranks=pd.Series(np.r_[p,n]).rank(method='average').to_numpy()
 return float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))

def sensitivity(df):
 base=df[df.defect=='baseline'];out=[]
 for d,p in MATCHING_PROBE.items():
  a=df[(df.defect==d)&(df.probe==p)];b=base[base.probe==p]
  m=a.merge(b,on=['domain','case_id','probe'],suffixes=('_d','_b'))
  if m.empty:
   out.append([d,p,0,*([math.nan]*6)]);continue
  diff=m.strain_d-m.strain_b;lo,hi=bootstrap_ci(diff)
  out.append([d,p,len(m),diff.mean(),lo,hi,dz(diff),perm_p(diff),auc([1]*len(m)+[0]*len(m),list(m.strain_d)+list(m.strain_b))])
 return pd.DataFrame(out,columns=['defect','matching_probe','n_pairs','mean_delta','ci_low','ci_high','cohen_dz','permutation_p','auroc'])

def delta_matrix(df):
 base=df[df.defect=='baseline'].groupby('probe').strain.mean()
 defects=[d for d in MATCHING_PROBE if d in set(df.defect)]
 probes=sorted(p for p in set(df.probe) if p not in {'unknown','baseline'})
 M=pd.DataFrame(index=defects,columns=probes,dtype=float)
 for d in defects:
  means=df[df.defect==d].groupby('probe').strain.mean()
  for p in probes:
   if p in means.index and p in base.index:M.loc[d,p]=means[p]-base[p]
 return M

def specificity(M):
 rows=[]
 for d,p in MATCHING_PROBE.items():
  if d not in M.index or p not in M.columns:continue
  r=M.loc[d].dropna()
  if r.empty:continue
  others=r.drop(labels=[p],errors='ignore');runner=others.max() if len(others) else np.nan
  rows.append([d,p,r[p],runner,r[p]-runner if np.isfinite(runner) else np.nan,r.idxmax(),r.idxmax()==p])
 return pd.DataFrame(rows,columns=['defect','expected_probe','expected_delta','largest_off_target_delta','localization_margin','argmax_probe','localized_correctly'])

def plot_heatmap(M,path):
 if M.empty:return
 fig,ax=plt.subplots(figsize=(8.5,4.8));arr=M.to_numpy(float);im=ax.imshow(arr,aspect='auto')
 ax.set_xticks(range(len(M.columns)),M.columns,rotation=30,ha='right');ax.set_yticks(range(len(M.index)),M.index)
 ax.set_xlabel('Probe transformation');ax.set_ylabel('Injected defect');ax.set_title('CAI-Bench localizes injected policy regressions\n(mean strain increase over baseline)')
 for i in range(arr.shape[0]):
  for j in range(arr.shape[1]):
   if np.isfinite(arr[i,j]):ax.text(j,i,f'{arr[i,j]:+.2f}',ha='center',va='center')
 fig.colorbar(im,ax=ax,label='Δ CAI Strain');fig.tight_layout();fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig)

def plot_joint(df,path):
 g=df.groupby('defect').agg(strain=('strain','mean'),helpfulness=('helpfulness','mean')).reset_index();g=g[np.isfinite(g.helpfulness)]
 if g.empty:return
 fig,ax=plt.subplots(figsize=(7.2,5.5));ax.scatter(g.strain,g.helpfulness,s=70)
 for _,r in g.iterrows():ax.annotate(r.defect,(r.strain,r.helpfulness),xytext=(5,5),textcoords='offset points',fontsize=9)
 ax.set_xlabel('Mean CAI Strain');ax.set_ylabel('Mean helpfulness');ax.set_title('Consistency and helpfulness are distinct');ax.set_ylim(-.05,1.05);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('results_json',type=Path);ap.add_argument('--out-dir',type=Path,default=Path('validity_outputs'));a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 df=normalize(json.loads(a.results_json.read_text()));df.to_csv(a.out_dir/'normalized_case_results.csv',index=False)
 s=sensitivity(df);s.to_csv(a.out_dir/'sensitivity_statistics.csv',index=False)
 M=delta_matrix(df);M.to_csv(a.out_dir/'defect_probe_delta_matrix.csv')
 sp=specificity(M);sp.to_csv(a.out_dir/'specificity_statistics.csv',index=False)
 plot_heatmap(M,a.out_dir/'figure_defect_localization_heatmap.png');plot_joint(df,a.out_dir/'figure_strain_vs_helpfulness.png')
 print('Loaded',len(df),'rows');print(s.to_string(index=False));print('\nSaved to',a.out_dir.resolve())
if __name__=='__main__':main()
