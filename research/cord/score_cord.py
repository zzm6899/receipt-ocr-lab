"""Cluster-paired CORD total exact accuracy and structured menu-row scoring."""
from __future__ import annotations
import csv,json,re,unicodedata,random,math
from pathlib import Path
from decimal import Decimal,InvalidOperation
import numpy as np
import sys
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from row_metrics import score_rows,donut_rows

ROOT=Path(__file__).resolve().parent
SPLIT="test"
VARIANTS=["clean","color_cast","fade","occlusion","rotate180","synthetic_crumple"]
SYSTEMS={"Tesseract+rules":("ocr",["nm","cnt","price"]),"LayoutLMv3":("layout",["nm","cnt","price"]),"TrOCR→LayoutLMv3":("cascade",["nm","cnt","price"]),"Donut":("donut",["nm","cnt","price"])}
OUT=ROOT/"cord_data"/"results"; OUT.mkdir(parents=True,exist_ok=True)

def textnorm(v):
    if v is None: return None
    s=unicodedata.normalize("NFKD",str(v)).casefold()
    return "".join(c for c in s if c.isalnum())
def numcanon(v):
    if v is None: return None
    ds="".join(re.findall(r"\d",str(v)))
    return ds.lstrip("0") or "0" if ds else None
def parse_amount(v):
    """Parse one signed amount; grouping separators have 3 digits, decimal has 1-2."""
    if v is None: return None
    s=re.sub(r"(?i)(?:IDR|RP|RM|USD|EUR)","",str(v)).strip()
    tokens=re.findall(r"(?<![A-Za-z0-9])[+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*(?![A-Za-z0-9])",s)
    tokens=[t.strip() for t in tokens if re.search(r"\d",t)]
    if len(tokens)!=1: return None
    token=re.sub(r"\s*([.,])\s*",r"\1",tokens[0]).replace(" ","")
    sign=-1 if token.startswith("-") else 1; token=token.lstrip("+-")
    if not token or ":" in token or "-" in token or not re.fullmatch(r"\d[\d.,]*",token): return None
    if "," in token and "." in token:
        dec=max(token.rfind(","),token.rfind(".")); sep=token[dec]; integer=token[:dec]; fraction=token[dec+1:]
        group_sep="." if sep=="," else ","
        chunks=integer.split(group_sep)
        if not fraction.isdigit() or len(fraction) not in (1,2) or not chunks[0].isdigit() or not 1<=len(chunks[0])<=3 or any(not c.isdigit() or len(c)!=3 for c in chunks[1:]): return None
        norm="".join(chunks)+"."+fraction
    elif "," in token or "." in token:
        sep="," if "," in token else "."; chunks=token.split(sep)
        if any(not c.isdigit() for c in chunks): return None
        if len(chunks)>2:
            if not 1<=len(chunks[0])<=3 or any(len(c)!=3 for c in chunks[1:]): return None
            norm="".join(chunks)
        elif len(chunks)==2 and len(chunks[1])==3:
            if not 1<=len(chunks[0])<=3: return None
            norm="".join(chunks)
        elif len(chunks)==2 and len(chunks[1]) in (1,2): norm=chunks[0]+"."+chunks[1]
        else: return None
    else: norm=token
    try: return Decimal(norm)*sign
    except InvalidOperation: return None
def parse_quantity(v):
    if v is None: return None
    return parse_amount(re.sub(r"(?i)x", "", str(v)))
def anls_total(pred, truth):
    """ST-VQA-style thresholded normalized edit similarity on canonical total digits."""
    a,b=numcanon(pred),numcanon(truth)
    if a is None or b is None: return 0.0
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    similarity=1-prev[-1]/max(1,len(a),len(b))
    return float(similarity) if similarity>0.5 else 0.0
def total_value(raw, system):
    if system=="Donut":
        parsed=raw.get("donut_json",{}) if isinstance(raw,dict) else {}
        t=parsed.get("total") if isinstance(parsed,dict) else None
        if isinstance(t,list): t=t[0] if t else None
        v=t.get("total_price") if isinstance(t,dict) else None
        return v if isinstance(v,(str,int,float)) else None
    if system=="Tesseract+rules": return raw.get("ocr_rules_total")
    key="layout_fields" if system=="LayoutLMv3" else "cascade_fields"
    v=(raw.get(key) or {}).get("total")
    return v if isinstance(v,(str,int,float)) else None
def gt_total(gt):
    t=gt.get("gt_parse",{}).get("total")
    if isinstance(t,list): t=t[0] if t else None
    return t.get("total_price") if isinstance(t,dict) else None
def menu_list(obj):
    x=obj.get("menu",[]) if isinstance(obj,dict) else []
    if isinstance(x,dict): return [x]
    return x if isinstance(x,list) else []
def gt_menu_rows(gt):
    """Use the structured parse items and cross-check one row per menu group_id."""
    parsed=menu_list(gt.get("gt_parse",{}))
    group_ids=sorted({line.get("group_id") for line in gt.get("valid_line",[]) if str(line.get("category","" )).startswith("menu.") and line.get("group_id") is not None})
    if len(parsed)!=len(group_ids):
        raise ValueError(f"CORD menu/group_id mismatch: parsed_rows={len(parsed)} groups={len(group_ids)}")
    return [{**row,"_gt_group_id":group_id} for row,group_id in zip(parsed,group_ids) if isinstance(row,dict)]
def pred_rows(raw, system):
    if system=="Tesseract+rules": return raw.get("ocr_rows",[])
    if system=="LayoutLMv3": return raw.get("layout_rows",[])
    if system=="TrOCR→LayoutLMv3": return raw.get("cascade_rows",[])
    return donut_rows(raw.get("donut_json",{}))
def menu_metrics(gt_rows,predicted,support):
    return score_rows(gt_rows,predicted,support)
def donut_valid(raw):
    p=raw.get("donut_json")
    return int(isinstance(p,dict))
def fmt(x): return "" if x is None else f"{x:.6f}"
def wilson(k,n,z=1.959963984540054):
    if n<=0: return 0.0,0.0
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return max(0.0,center-half),min(1.0,center+half)

manifest=json.loads((ROOT/"cord_data"/SPLIT/"manifest.json").read_text(encoding="utf-8"))
ids=[r["receipt_id"] for r in manifest]
records={}
missing=[]
for rid in ids:
    gt=json.loads((ROOT/"cord_data"/SPLIT/f"{rid}.json").read_text(encoding="utf-8"))["ground_truth"]
    records[rid]={"gt":gt,"pred":{}}
    for v in VARIANTS:
        p=ROOT/"cord_data"/"direct_predictions"/SPLIT/f"{rid}__{v}.json"
        if not p.exists(): missing.append((rid,v)); continue
        records[rid]["pred"][v]=json.loads(p.read_text(encoding="utf-8"))

N=len(ids); rng=np.random.default_rng(20260930); boot=rng.integers(0,N,size=(10000,N))
metrics=[]
for system,(rowkey,support) in SYSTEMS.items():
    for variant in VARIANTS:
        corr=[]; digits_match=[]; eligible_mask=[]; rowcounts=[]; valid=[]; total_cov=[]; anls_scores=[]; amount_valid=[]
        for rid in ids:
            info=records[rid]; raw=info["pred"].get(variant,{})
            truth=gt_total(info["gt"]); pred=total_value(raw,system)
            eligible_mask.append(int(truth is not None))
            corr.append(int(truth is not None and parse_amount(pred) is not None and parse_amount(pred)==parse_amount(truth)))
            digits_match.append(int(truth is not None and numcanon(pred)==numcanon(truth)))
            amount_valid.append(int(parse_amount(pred) is not None))
            anls_scores.append(anls_total(pred,truth) if truth is not None else 0.0)
            total_cov.append(int(pred not in (None,"")))
            gtrows=gt_menu_rows(info["gt"])
            if system=="Donut": predicted=pred_rows(raw,system)
            else: predicted=raw.get(rowkey+"_rows",[])
            rowcounts.append(menu_metrics(gtrows,predicted,support))
            valid.append(donut_valid(raw) if system=="Donut" else int(bool(raw)))
        x=np.asarray(corr,dtype=float); elig=np.asarray(eligible_mask,dtype=float); av=np.asarray(anls_scores,dtype=float)
        denom=elig[boot].sum(axis=1)
        samp=np.divide((x*elig)[boot].sum(axis=1),denom,out=np.zeros(len(denom)),where=denom>0)
        anls_boot=np.divide((av*elig)[boot].sum(axis=1),denom,out=np.zeros(len(denom)),where=denom>0)
        anls_lo,anls_hi=np.quantile(anls_boot,[.025,.975])
        acc_k=int((x*elig).sum()); acc_n=int(elig.sum()); lo,hi=wilson(acc_k,acc_n); acc=acc_k/max(1,acc_n)
        dm=np.asarray(digits_match,dtype=float); digit_k=int((dm*elig).sum()); digit_lo,digit_hi=wilson(digit_k,acc_n)
        clean=np.asarray([int(gt_total(records[r]["gt"]) is not None and parse_amount(total_value(records[r]["pred"].get("clean",{}),system)) is not None and parse_amount(total_value(records[r]["pred"].get("clean",{}),system))==parse_amount(gt_total(records[r]["gt"]))) for r in ids],dtype=float)
        d=(x-clean)*elig; db=np.divide(d[boot].sum(axis=1),denom,out=np.zeros(len(denom)),where=denom>0); dlo,dhi=np.quantile(db,[.025,.975])
        cnt=np.asarray(rowcounts,dtype=int); sums=cnt[boot].sum(axis=1)
        tp,fp,fn=np.asarray(cnt).sum(axis=0)[:3]
        f1=(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0
        f1boot=np.divide(2*sums[:,0],2*sums[:,0]+sums[:,1]+sums[:,2],out=np.zeros(len(sums)),where=(2*sums[:,0]+sums[:,1]+sums[:,2])>0)
        flo,fhi=np.quantile(f1boot,[.025,.975])
        clean_counts=[]
        for rid in ids:
            info=records[rid]; raw=info["pred"].get("clean",{})
            gtrows=gt_menu_rows(info["gt"])
            predicted=pred_rows(raw,system) if system=="Donut" else raw.get(rowkey+"_rows",[])
            clean_counts.append(menu_metrics(gtrows,predicted,support))
        cc=np.asarray(clean_counts,dtype=int); cs=cc[boot].sum(axis=1)
        clean_f1boot=np.divide(2*cs[:,0],2*cs[:,0]+cs[:,1]+cs[:,2],out=np.zeros(len(cs)),where=(2*cs[:,0]+cs[:,1]+cs[:,2])>0)
        df1boot=f1boot-clean_f1boot; df1lo,df1hi=np.quantile(df1boot,[.025,.975])
        df1=float(f1-(2*int(cc[:,0].sum())/(2*int(cc[:,0].sum())+int(cc[:,1].sum())+int(cc[:,2].sum())) if (2*int(cc[:,0].sum())+int(cc[:,1].sum())+int(cc[:,2].sum())) else 0))
        row_recall=tp/max(1,int(cnt[:,3].sum()))
        row_precision=tp/max(1,int(cnt[:,4].sum()))
        complete=(cnt[:,0]==cnt[:,3])&(cnt[:,1]==0)&(cnt[:,2]==0)&(cnt[:,3]>0)
        eligible_receipts=cnt[:,3]>0; crden=eligible_receipts[boot].sum(axis=1)
        crboot=np.divide(complete[boot].sum(axis=1),crden,out=np.zeros(len(crden)),where=crden>0); crlo,crhi=np.quantile(crboot,[.025,.975])
        metrics.append({"system":system,"variant":variant,"receipts":N,"total_gt_n":acc_n,"total_predictions_present":int(np.sum(np.asarray(total_cov)*elig)),"total_prediction_coverage":float(np.sum(np.asarray(total_cov)*elig)/max(1,elig.sum())),"total_exact_correct":acc_k,"total_exact_accuracy":acc,"total_exact_ci95_low":float(lo),"total_exact_ci95_high":float(hi),"total_digits_only_match_correct":digit_k,"total_digits_only_match_accuracy":digit_k/max(1,acc_n),"total_digits_only_match_ci95_low":float(digit_lo),"total_digits_only_match_ci95_high":float(digit_hi),"total_amount_valid_predictions":int(np.sum(np.asarray(amount_valid)*elig)),"total_invalid_or_ambiguous_predictions":int(acc_n-int(np.sum(np.asarray(amount_valid)*elig))),"total_anls":float((av*elig).sum()/max(1,elig.sum())),"total_anls_ci95_low":float(anls_lo),"total_anls_ci95_high":float(anls_hi),"delta_pp_vs_clean":float(d.sum()/max(1,elig.sum())*100),"delta_pp_ci95_low":float(dlo*100),"delta_pp_ci95_high":float(dhi*100),"menu_gt_rows_eligible":int(cnt[:,3].sum()),"menu_pred_rows":int(cnt[:,4].sum()),"menu_rows_tp":int(tp),"menu_rows_fp":int(fp),"menu_rows_fn":int(fn),"menu_row_precision":row_precision,"menu_row_recall":row_recall,"menu_row_f1":float(f1),"menu_row_f1_ci95_low":float(flo),"menu_row_f1_ci95_high":float(fhi),"menu_row_f1_delta_vs_clean":df1,"menu_row_f1_delta_ci95_low":float(df1lo),"menu_row_f1_delta_ci95_high":float(df1hi),"all_items_correct_receipts":int(complete.sum()),"all_items_correct_receipt_n":int(eligible_receipts.sum()),"all_items_correct_receipt_rate":float(complete.sum()/max(1,eligible_receipts.sum())),"all_items_correct_receipt_ci95_low":float(crlo),"all_items_correct_receipt_ci95_high":float(crhi),"donut_json_parse_rate":float(np.mean(valid)) if system=="Donut" else None,"failed_prediction_files":sum((v not in records[r]["pred"]) for r in ids for v in [variant])})

with (OUT/"metrics_by_system_condition.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
with (OUT/"missing_predictions.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["receipt_id","variant"]);w.writerows(missing)

comparisons=[]
for variant in VARIANTS:
    for i,a in enumerate(SYSTEMS):
        for b in list(SYSTEMS)[i+1:]:
            eligible=np.asarray([int(gt_total(records[r]["gt"]) is not None) for r in ids],dtype=float)
            pa=np.asarray([int(gt_total(records[r]["gt"]) is not None and parse_amount(total_value(records[r]["pred"].get(variant,{}),a)) is not None and parse_amount(total_value(records[r]["pred"].get(variant,{}),a))==parse_amount(gt_total(records[r]["gt"]))) for r in ids],dtype=float)
            pb=np.asarray([int(gt_total(records[r]["gt"]) is not None and parse_amount(total_value(records[r]["pred"].get(variant,{}),b)) is not None and parse_amount(total_value(records[r]["pred"].get(variant,{}),b))==parse_amount(gt_total(records[r]["gt"]))) for r in ids],dtype=float)
            denom=eligible[boot].sum(axis=1); delta=pa-pb
            bootdelta=np.divide(delta[boot].sum(axis=1),denom,out=np.zeros(len(denom)),where=denom>0)
            lo,hi=np.quantile(bootdelta,[.025,.975])
            comparisons.append({"variant":variant,"system_a":a,"system_b":b,"n_total_gt":int(eligible.sum()),"correct_a":int(pa.sum()),"correct_b":int(pb.sum()),"difference_a_minus_b_pp":float(delta.sum()/max(1,eligible.sum())*100),"paired_bootstrap_ci95_low_pp":float(lo*100),"paired_bootstrap_ci95_high_pp":float(hi*100)})
with (OUT/"paired_model_differences.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)

# Joint paired receipt-ID bootstrap comparisons, identical resamples for every system and condition.
by={(m["system"],m["variant"]):m for m in metrics}
print(json.dumps({"n_receipts":N,"missing_prediction_cells":len(missing),"systems":len(SYSTEMS),"conditions":len(VARIANTS),"results_csv":str(OUT/"metrics_by_system_condition.csv")},indent=2))
for s in SYSTEMS:
 print(s,[(v,round(by[(s,v)]["total_exact_accuracy"],3),round(by[(s,v)]["menu_row_f1"],3),round(by[(s,v)]["delta_pp_vs_clean"],1)) for v in VARIANTS])
