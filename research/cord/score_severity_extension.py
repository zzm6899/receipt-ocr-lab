"""Score frozen high-severity extension against the matching clean images."""
from __future__ import annotations
import csv, json, re, sys, math
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
from row_metrics import score_rows, donut_rows

ROOT=Path(__file__).resolve().parent
SPLIT='test'; PRED=ROOT/'cord_data'/'results'/f'severity_extension_{SPLIT}'
BASE=ROOT/'cord_data'/'results'/f'severity_{SPLIT}'
GTROOT=ROOT/'cord_data'/SPLIT
SYSTEMS=('Tesseract+rules','LayoutLMv3','TrOCR→LayoutLMv3','Donut')
CONDITIONS=('blue_cast_high','blue_cast_extreme','fade_extreme','fade_maximum',
            'occlusion_28','occlusion_38','blue_fade_mix','fade_occlusion_mix')

def parse_amount(v):
    if v is None:return None
    s=re.sub(r'(?i)(?:IDR|RP|RM|USD|EUR|AUD|NZD|SGD|MYR|\$|€|£|¥)','',str(v)).strip()
    tokens=re.findall(r'(?<![A-Za-z0-9])[+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*(?![A-Za-z0-9])',s)
    if len(tokens)!=1:return None
    token=re.sub(r'\s*([.,])\s*',r'\1',tokens[0]).replace(' ','')
    sign=-1 if token.startswith('-') else 1;token=token.lstrip('+-')
    if not token or not re.fullmatch(r'\d[\d.,]*',token):return None
    if ',' in token and '.' in token:
        dec=max(token.rfind(','),token.rfind('.'));sep=token[dec]
        chunks=token[:dec].split('.' if sep==',' else ',');frac=token[dec+1:]
        if not frac.isdigit() or len(frac) not in (1,2) or not 1<=len(chunks[0])<=3 or any(not x.isdigit() or len(x)!=3 for x in chunks[1:]):return None
        norm=''.join(chunks)+'.'+frac
    elif ',' in token or '.' in token:
        sep=',' if ',' in token else '.';chunks=token.split(sep)
        if any(not x.isdigit() for x in chunks):return None
        if len(chunks)>2:
            if not 1<=len(chunks[0])<=3 or any(len(x)!=3 for x in chunks[1:]):return None
            norm=''.join(chunks)
        elif len(chunks)==2 and len(chunks[1])==3 and 1<=len(chunks[0])<=3:norm=''.join(chunks)
        elif len(chunks)==2 and len(chunks[1]) in (1,2):norm=chunks[0]+'.'+chunks[1]
        else:return None
    else:norm=token
    try:return Decimal(norm)*sign
    except InvalidOperation:return None

def gt_total(gt):
    x=(gt.get('gt_parse') or {}).get('total')
    if isinstance(x,list):x=x[0] if x else None
    return x.get('total_price') if isinstance(x,dict) else None

def donut_total(parsed):
    nodes=parsed if isinstance(parsed,list) else [parsed]
    for node in nodes:
        if not isinstance(node,dict):continue
        val=node.get('total')
        vals=val if isinstance(val,list) else [val]
        for item in vals:
            if isinstance(item,dict) and item.get('total_price') not in (None,''):return item['total_price']
            if isinstance(item,(str,int,float)) and 'total' in node:return item
        if node.get('total_price') not in (None,''):return node['total_price']
    return None

def prediction_total(raw,system):
    if system=='Tesseract+rules':return raw.get('ocr_rules_total')
    if system=='LayoutLMv3':return (raw.get('layout_fields') or {}).get('total')
    if system=='TrOCR→LayoutLMv3':return (raw.get('cascade_fields') or {}).get('total')
    return donut_total(raw.get('donut_json'))

def prediction_rows(raw,system):
    if system=='Tesseract+rules':return raw.get('ocr_rows',[])
    if system=='LayoutLMv3':return raw.get('layout_rows',[])
    if system=='TrOCR→LayoutLMv3':return raw.get('cascade_rows',[])
    return donut_rows(raw.get('donut_json'))

def menu_gt(gt):
    parsed=(gt.get('gt_parse') or {}).get('menu',[])
    rows=[parsed] if isinstance(parsed,dict) else parsed if isinstance(parsed,list) else []
    group_ids={line.get('group_id') for line in gt.get('valid_line',[]) if str(line.get('category','')).startswith('menu.') and line.get('group_id') is not None}
    if len(rows)!=len(group_ids):raise ValueError(f'CORD menu/group_id mismatch: parsed={len(rows)}, groups={len(group_ids)}')
    return rows

def f1(cnt):
    tp,fp,fn=cnt[:3];return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

def wilson(k,n,z=1.959963984540054):
    if n<=0:return 0.,0.
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)

def main():
    ids=[x['receipt_id'] for x in json.loads((GTROOT/'manifest.json').read_text(encoding='utf-8'))]
    if len(ids)!=100:raise ValueError(f'Expected frozen 100 test IDs; got {len(ids)}')
    rng=np.random.default_rng(20261001);boot=rng.integers(0,len(ids),size=(10000,len(ids)))
    cells=[];image_manifest={(r['receipt_id'],r['condition']):r for r in csv.DictReader((PRED/'image_manifest.csv').open(encoding='utf-8'))}
    expected_cells={(rid,cond) for rid in ids for cond in CONDITIONS}
    if set(image_manifest)!=expected_cells:raise ValueError('Image manifest does not contain exactly 800 frozen receipt-condition pairs')
    published_clean={r['model']:r for r in csv.DictReader((BASE/'severity_metrics.csv').open(encoding='utf-8')) if r['condition']=='clean'}
    core_names={'TrOCR→LayoutLMv3':'TrOCR→LayoutLMv3'}
    for system in SYSTEMS:
        clean={}
        for rid in ids:
            gt=json.loads((GTROOT/f'{rid}.json').read_text(encoding='utf-8'))['ground_truth']
            raw=json.loads((BASE/f'{rid}__clean.json').read_text(encoding='utf-8'))
            source=image_manifest[(rid,CONDITIONS[0])]['source_sha256']
            if raw.get('variant_sha256')!=source:raise ValueError(f'Clean comparator hash mismatch for {rid}: expected {source}, got {raw.get("variant_sha256")}')
            truth=gt_total(gt);p=parse_amount(prediction_total(raw,system));t=parse_amount(truth)
            eligible=t is not None
            rows=score_rows(menu_gt(gt),prediction_rows(raw,system))[:3]
            clean[rid]={'correct':int(eligible and p is not None and p==t),'eligible':int(eligible),'rows':rows}
        crows=np.array([clean[r]['rows'] for r in ids],dtype=int);ctp,cfp,cfn=crows.sum(axis=0)
        our_clean_f1=f1((ctp,cfp,cfn)); prior=published_clean[core_names.get(system,system)]
        if abs(our_clean_f1-float(prior['menu_row_f1']))>1e-12:
            raise ValueError(f'Clean row scorer mismatch for {system}: {our_clean_f1} vs {prior["menu_row_f1"]}')
        clean_total=sum(x['correct'] for x in clean.values());prior_total=int(prior['total_exact_correct'])
        if clean_total!=prior_total:
            raise ValueError(f'Clean total scorer mismatch for {system}: {clean_total} vs {prior_total}')
        for cond in CONDITIONS:
            stats=[]
            for rid in ids:
                gt=json.loads((GTROOT/f'{rid}.json').read_text(encoding='utf-8'))['ground_truth']
                raw=json.loads((PRED/f'{rid}__{cond}.json').read_text(encoding='utf-8'))
                expected_sha=image_manifest[(rid,cond)]['variant_sha256']
                if raw.get('variant_sha256')!=expected_sha:raise ValueError(f'Variant hash mismatch for {rid}/{cond}')
                if raw.get('ground_truth_inference_prohibited') is not True:raise ValueError(f'Inference provenance flag missing for {rid}/{cond}')
                raw_total=prediction_total(raw,system);truth=gt_total(gt);p=parse_amount(raw_total);t=parse_amount(truth)
                eligible=t is not None;rows=score_rows(menu_gt(gt),prediction_rows(raw,system))[:3]
                donutparsed=raw.get('donut_json')
                stats.append({'correct':int(eligible and p is not None and p==t),'eligible':int(eligible),'rows':rows,
                    'total_present':int(raw_total not in (None,'')),'total_numeric_valid':int(p is not None),
                    'donut_json_valid':int(isinstance(donutparsed,(dict,list))) if system=='Donut' else None})
            elig=np.array([x['eligible'] for x in stats],dtype=int)
            ca=np.array([x['correct'] for x in stats],dtype=float);cb=np.array([x['correct'] for x in clean.values()],dtype=float)
            denominator=elig[boot].sum(axis=1);delta=(ca-cb)*elig
            bootdelta=np.divide(delta[boot].sum(axis=1),denominator,out=np.zeros(10000),where=denominator>0)
            lo,hi=np.quantile(bootdelta,[.025,.975])
            cc=np.array([x['rows'] for x in stats],dtype=int);cl=np.array([clean[r]['rows'] for r in ids],dtype=int)
            samp=cc[boot].sum(axis=1);cleansamp=cl[boot].sum(axis=1)
            fboots=np.divide(2*samp[:,0],2*samp[:,0]+samp[:,1]+samp[:,2],out=np.zeros(10000),where=(2*samp[:,0]+samp[:,1]+samp[:,2])>0)
            dfboots=fboots-np.divide(2*cleansamp[:,0],2*cleansamp[:,0]+cleansamp[:,1]+cleansamp[:,2],out=np.zeros(10000),where=(2*cleansamp[:,0]+cleansamp[:,1]+cleansamp[:,2])>0)
            tp,fp,fn=cc.sum(axis=0);ctp,cfp,cfn=cl.sum(axis=0)
            fl,fh=np.quantile(fboots,[.025,.975]);dl,dh=np.quantile(dfboots,[.025,.975]);tlo,thi=wilson(int(ca.sum()),int(elig.sum()))
            cells.append({'system':system,'condition':cond,'receipts':len(ids),'total_gt_n':int(elig.sum()),
                'total_exact_correct':int(ca.sum()),'total_exact_accuracy':float(ca.sum()/max(1,elig.sum())),
                'total_exact_wilson95_low':float(tlo),'total_exact_wilson95_high':float(thi),
                'total_prediction_present_n':sum(x['total_present'] for x in stats),
                'total_numeric_valid_n':sum(x['total_numeric_valid'] for x in stats),
                'donut_json_valid_n':sum(x['donut_json_valid'] or 0 for x in stats) if system=='Donut' else None,
                'clean_total_exact_correct':int(cb.sum()),'clean_total_exact_accuracy':float(cb.sum()/max(1,elig.sum())),
                'total_delta_pp_vs_clean':float((delta.sum()/max(1,elig.sum()))*100),
                'total_delta_paired_bootstrap95_low_pp':float(lo*100),'total_delta_paired_bootstrap95_high_pp':float(hi*100),
                'menu_gt_rows':int(tp+fn),'menu_pred_rows':int(tp+fp),'menu_tp':int(tp),'menu_fp':int(fp),'menu_fn':int(fn),
                'menu_row_f1':f1((tp,fp,fn)),'menu_row_f1_ci95_low':float(fl),'menu_row_f1_ci95_high':float(fh),
                'clean_menu_row_f1':f1((ctp,cfp,cfn)),'menu_row_f1_delta_vs_clean_pp':float((f1((tp,fp,fn))-f1((ctp,cfp,cfn)))*100),
                'menu_row_f1_delta_bootstrap95_low_pp':float(dl*100),'menu_row_f1_delta_bootstrap95_high_pp':float(dh*100)})
    path=PRED/'metrics.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    run_path=PRED/'run_manifest.json';run=json.loads(run_path.read_text(encoding='utf-8'))
    run['completed_prediction_cells']=len(expected_cells)
    run['all_prediction_cells_hash_validated']=True
    run_path.write_text(json.dumps(run,indent=2),encoding='utf-8')
    (PRED/'scoring_manifest.json').write_text(json.dumps({'split':'official CORD v2 test','receipt_ids':len(ids),
        'conditions':CONDITIONS,'systems':SYSTEMS,'bootstrap_unit':'original CORD receipt ID; paired clean-versus-corrupt resampling, 10,000 draws, NumPy seed 20261001',
        'totals':'strict signed decimal-preserving one-amount parser; no ANLS for primary result',
        'rows':'canonical shared row_metrics.score_rows; one-to-one maximum bipartite matching with CORD quantity-unit policy',
        'clean_reference':'existing severity_test clean predictions for the same exact severity_test clean JPEGs that seeded extension variants',
        'crosscheck':'Each system clean exact-total count and canonical menu row F1 must exactly match the existing severity_test aggregate before condition deltas are reported.',
        'note':'Synthetic severe variants only; no claims about physical damage. Clean image and each condition are paired by receipt ID.'},indent=2),encoding='utf-8')
    print(f'Wrote {len(cells)} system-condition cells across {len(ids)} test receipt IDs: {path}')

if __name__=='__main__':main()
