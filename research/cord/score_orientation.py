"""Paired held-out CORD scoring of the validation-selected orientation ablation."""
import csv,json,random,re,unicodedata
from decimal import Decimal,InvalidOperation
from pathlib import Path
from row_metrics import score_rows,donut_rows,normalize_menu
ROOT=Path(__file__).resolve().parent;RESULT=ROOT/'cord_data'/'results'/'orientation';
BASE=ROOT/'cord_data'/'direct_predictions'/'test';CORR=BASE/'../test_orientation_corrected'
SYSTEMS={'Tesseract+rules':'ocr','LayoutLMv3':'layout','TrOCR→LayoutLMv3':'cascade','Donut':'donut'}
CONDS=['clean','color_cast','fade','occlusion','rotate180','synthetic_crumple']
def amount(v):
    if v is None:return None
    s=re.sub(r'(?i)(?:IDR|RP|RM|USD|EUR)','',str(v)).strip()
    t=re.findall(r'(?<![A-Za-z0-9])[+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*(?![A-Za-z0-9])',s)
    if len(t)!=1:return None
    z=re.sub(r'\s*([.,])\s*',r'\1',t[0]).replace(' ','');sg=-1 if z.startswith('-') else 1;z=z.lstrip('+-')
    if not z or not re.fullmatch(r'\d[\d.,]*',z):return None
    if ',' in z and '.' in z:
        dec=max(z.rfind(','),z.rfind('.'));sep=z[dec];parts=z[:dec].split('.' if sep==',' else ',');frac=z[dec+1:]
        if not frac.isdigit() or len(frac) not in (1,2) or not 1<=len(parts[0])<=3 or any(len(x)!=3 or not x.isdigit() for x in parts[1:]):return None
        n=''.join(parts)+'.'+frac
    elif ',' in z or '.' in z:
        sep=',' if ',' in z else '.';parts=z.split(sep)
        if any(not x.isdigit() for x in parts):return None
        if len(parts)>2:
            if not 1<=len(parts[0])<=3 or any(len(x)!=3 for x in parts[1:]):return None
            n=''.join(parts)
        elif len(parts)==2 and len(parts[1])==3 and 1<=len(parts[0])<=3:n=''.join(parts)
        elif len(parts)==2 and len(parts[1]) in (1,2):n=parts[0]+'.'+parts[1]
        else:return None
    else:n=z
    try:return Decimal(n)*sg
    except InvalidOperation:return None
def txt(v):return ''.join(c for c in unicodedata.normalize('NFKD',str(v or '')).casefold() if c.isalnum())
def qty(v):
    if v is None:return None
    s=re.sub(r'(?i)x','',str(v));n=amount(s)
    if n is None:return None
    toks=re.findall(r'(?<![A-Za-z0-9])[+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*(?![A-Za-z0-9])',s)
    if len(toks)!=1:return None
    unit=re.sub(re.escape(toks[0]),'',s,count=1).strip().casefold()
    return n,re.sub(r'\s+',' ',unit)
def total(gt,raw,sys):
    if sys=='Donut':
        parsed=raw.get('donut_json') or {}
        if isinstance(parsed,list): return None
        t=parsed.get('total');t=t[0] if isinstance(t,list) and t else t
        return t.get('total_price') if isinstance(t,dict) else None
    return (raw.get({'Tesseract+rules':'ocr_rules_total','LayoutLMv3':'layout_fields','TrOCR→LayoutLMv3':'cascade_fields'}[sys]) if sys=='Tesseract+rules' else None) if sys=='Tesseract+rules' else (raw.get({'LayoutLMv3':'layout_fields','TrOCR→LayoutLMv3':'cascade_fields'}[sys]) or {}).get('total')
def gt_total(gt):
    t=(gt.get('gt_parse') or {}).get('total');t=t[0] if isinstance(t,list) and t else t
    return t.get('total_price') if isinstance(t,dict) else None
def menu(x):
    return normalize_menu(x)
def row_counts(gt,pred):
    return score_rows(menu(gt.get('gt_parse')),pred)[:3]
def preds(raw,sys):
    if sys=='Donut':return donut_rows(raw.get('donut_json'))
    return raw.get({'Tesseract+rules':'ocr_rows','LayoutLMv3':'layout_rows','TrOCR→LayoutLMv3':'cascade_rows'}[sys],[])
def f1(c):
    tp,fp,fn=c;return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
def boot_ci(arr,seed=7231,B=10000):
    rr=random.Random(seed);n=len(arr);values=[sum(arr[rr.randrange(n)] for _ in range(n))/n for _ in range(B)];values.sort();return values[int(.025*B)],values[int(.975*B)]
def main():
    ids=[x['receipt_id'] for x in json.loads((ROOT/'cord_data'/'test'/'manifest.json').read_text(encoding='utf-8'))]
    results=[];per={}
    for sys in SYSTEMS:
        for cond in CONDS:
            entries=[]
            for rid in ids:
                gt=json.loads((ROOT/'cord_data'/'test'/f'{rid}.json').read_text(encoding='utf-8'))['ground_truth']
                a=json.loads((BASE/f'{rid}__{cond}.json').read_text(encoding='utf-8'))
                b=json.loads((CORR/f'{rid}__{cond}.json').resolve().read_text(encoding='utf-8'))
                truth=gt_total(gt);eligible=truth is not None
                ca=int(eligible and amount(total(gt,a,sys)) is not None and amount(total(gt,a,sys))==amount(truth))
                cb=int(eligible and amount(total(gt,b,sys)) is not None and amount(total(gt,b,sys))==amount(truth))
                ra=row_counts(gt,preds(a,sys));rb=row_counts(gt,preds(b,sys))
                entries.append({'receipt_id':rid,'ta':ca,'tb':cb,'eligible':int(eligible),'ra':ra,'rb':rb,
                    'flipped':int(bool(b.get('orientation_was_corrected'))),'menu_rows':len(menu(gt.get('gt_parse')))} )
            per[(sys,cond)]=entries
            n=sum(x['eligible'] for x in entries);ka=sum(x['ta'] for x in entries);kb=sum(x['tb'] for x in entries)
            d=[x['tb']-x['ta'] for x in entries if x['eligible']];lo,hi=boot_ci(d)
            ca=tuple(sum(x['ra'][i] for x in entries) for i in range(3));cb=tuple(sum(x['rb'][i] for x in entries) for i in range(3))
            rr=random.Random(8162);boots=[];ids0=[x['receipt_id'] for x in entries];lookup={x['receipt_id']:x for x in entries}
            for _ in range(10000):
                sample=[lookup[ids0[rr.randrange(len(ids0))]] for _ in ids0]
                ac=tuple(sum(x['ra'][i] for x in sample) for i in range(3));bc=tuple(sum(x['rb'][i] for x in sample) for i in range(3));boots.append(f1(bc)-f1(ac))
            boots.sort();fa,fb=f1(ca),f1(cb)
            results.append({'model':sys,'condition':cond,'receipts':len(ids),'total_gt_n':n,'baseline_total_correct':ka,'corrected_total_correct':kb,
              'baseline_total_accuracy':ka/max(1,n),'corrected_total_accuracy':kb/max(1,n),'correction_delta_pp':(kb-ka)/max(1,n)*100,
              'total_delta_paired_bootstrap95_low_pp':lo*100,'total_delta_paired_bootstrap95_high_pp':hi*100,
              'flipped_images':sum(x['flipped'] for x in entries),'baseline_row_tp':ca[0],'baseline_row_fp':ca[1],'baseline_row_fn':ca[2],
              'corrected_row_tp':cb[0],'corrected_row_fp':cb[1],'corrected_row_fn':cb[2],
              'baseline_row_f1':fa,'corrected_row_f1':fb,'row_f1_delta_pp':(fb-fa)*100,
              'row_f1_delta_bootstrap95_low_pp':boots[int(.025*len(boots))]*100,'row_f1_delta_bootstrap95_high_pp':boots[int(.975*len(boots))]*100})
    with (RESULT/'test_orientation_model_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(f"wrote {len(results)} system-condition comparisons; n test IDs={len(ids)}")
if __name__=='__main__':main()
