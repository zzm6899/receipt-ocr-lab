"""Paired receipt bootstrap scorer for synthetic validation severity matrix."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import score_orientation as sc
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],default='validation');ap.add_argument('--limit',type=int);args=ap.parse_args()
OUT=ROOT/'cord_data'/'results'/f'severity_{args.split}';PRED=OUT;B=10000
conditions=['clean']+[f'{f}_{s}' for f in ('fade','blur','occlusion','lighting','crumple') for s in ('mild','moderate','severe')]
ids=[x['receipt_id'] for x in json.loads((ROOT/'cord_data'/args.split/'manifest.json').read_text(encoding='utf-8'))]
if args.limit is not None: ids=ids[:args.limit]
rng=np.random.default_rng(1729);boots=rng.integers(0,len(ids),size=(B,len(ids)))
metrics=[]
for system in sc.SYSTEMS:
    docs={}
    for rid in ids:
        gt=json.loads((ROOT/'cord_data'/args.split/f'{rid}.json').read_text(encoding='utf-8'))['ground_truth']
        docs[rid]={'gt':gt,'total':sc.gt_total(gt)}
        for cond in conditions:
            raw=json.loads((PRED/f'{rid}__{cond}.json').read_text(encoding='utf-8'))
            ptotal=sc.total(gt,raw,system);truth=sc.gt_total(gt);eligible=truth is not None
            correct=int(eligible and sc.amount(ptotal) is not None and sc.amount(ptotal)==sc.amount(truth))
            tp,fp,fn=sc.row_counts(gt,sc.preds(raw,system))
            docs[rid][cond]={'correct':correct,'eligible':int(eligible),'tp':tp,'fp':fp,'fn':fn,
              'total_present':int(ptotal not in (None,'')),'total_parseable':int(sc.amount(ptotal) is not None)}
    baseline=np.array([docs[r]['clean']['correct'] for r in ids],float)
    eligible=np.array([docs[r]['clean']['eligible'] for r in ids],float)
    clean_counts=np.array([[docs[r]['clean'][k] for k in ('tp','fp','fn')] for r in ids],int)
    clean_sample=clean_counts[boots].sum(axis=1)
    clean_f1=np.divide(2*clean_sample[:,0],2*clean_sample[:,0]+clean_sample[:,1]+clean_sample[:,2],out=np.zeros(B),where=2*clean_sample[:,0]+clean_sample[:,1]+clean_sample[:,2]>0)
    for cond in conditions:
        x=np.array([docs[r][cond]['correct'] for r in ids],float);elig=np.array([docs[r][cond]['eligible'] for r in ids],float)
        n=int(elig.sum());k=int((x*elig).sum());acc=k/max(1,n)
        d=(x-baseline)*elig;den=elig[boots].sum(axis=1);db=np.divide(d[boots].sum(axis=1),den,out=np.zeros(B),where=den>0);lo,hi=np.quantile(db,[.025,.975])
        counts=np.array([[docs[r][cond][q] for q in ('tp','fp','fn')] for r in ids],int);samp=counts[boots].sum(axis=1)
        f1=np.divide(2*samp[:,0],2*samp[:,0]+samp[:,1]+samp[:,2],out=np.zeros(B),where=2*samp[:,0]+samp[:,1]+samp[:,2]>0)
        rawtp,rawfp,rawfn=counts.sum(axis=0);rawf1=2*rawtp/(2*rawtp+rawfp+rawfn) if 2*rawtp+rawfp+rawfn else 0.
        # Joint paired difference from clean using the identical receipt-index matrix.
        fdelta=f1-clean_f1;flo,fhi=np.quantile(fdelta,[.025,.975])
        records=[docs[r][cond] for r in ids]
        metrics.append({'model':system,'condition':cond,'receipt_ids':len(ids),'total_gt_n':n,'total_exact_correct':k,
          'total_exact_accuracy':acc,'total_present_n':sum(z['total_present'] for z in records),
          'total_parseable_n':sum(z['total_parseable'] for z in records),
          'delta_pp_vs_clean':float(d.sum()/max(1,n)*100),'delta_pp_ci95_low':float(lo*100),'delta_pp_ci95_high':float(hi*100),
          'menu_gt_rows':int(rawtp+rawfn),'menu_pred_rows':int(rawtp+rawfp),'menu_tp':int(rawtp),'menu_fp':int(rawfp),'menu_fn':int(rawfn),
          'menu_row_f1':rawf1,'menu_row_f1_ci95_low':float(np.quantile(f1,.025)),'menu_row_f1_ci95_high':float(np.quantile(f1,.975)),
          'menu_row_f1_delta_vs_clean_pp':float((rawf1-(2*clean_counts[:,0].sum()/(2*clean_counts[:,0].sum()+clean_counts[:,1].sum()+clean_counts[:,2].sum()) if (2*clean_counts[:,0].sum()+clean_counts[:,1].sum()+clean_counts[:,2].sum()) else 0))*100),
          'menu_row_f1_delta_ci95_low_pp':float(flo*100),'menu_row_f1_delta_ci95_high_pp':float(fhi*100)})
with (OUT/'severity_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print(f"scored {len(metrics)} model-condition cells across {len(ids)} {args.split} receipt IDs; output={OUT/'severity_metrics.csv'}")
