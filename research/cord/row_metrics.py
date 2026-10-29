"""Shared, deterministic CORD menu-row normalization and one-to-one scoring.

Primary row schema is description (nm), quantity (cnt), and item total (price).
Only supported nonempty ground-truth fields are required for that row. Quantity
unit suffixes are preserved; the literal x marker in CORD values such as x1/1X
is normalized away. Every emitted nonempty CORD menu object is a candidate row;
objects with only unsupported menu fields count as unmatched false positives.
Empty/non-menu objects are ignored. Duplicate candidate/target rows are matched
maximum-bipartite one-to-one.
"""
from __future__ import annotations
import re, unicodedata
from decimal import Decimal, InvalidOperation

COMMON_FIELDS=("nm","cnt","price")
ROW_FIELDS=frozenset(("nm","cnt","price","unitprice","itemsubtotal"))

def text_norm(value):
    if value is None:return None
    s=unicodedata.normalize("NFKD",str(value)).casefold()
    return "".join(c for c in s if c.isalnum()) or None

def parse_amount(value):
    if value is None:return None
    s=re.sub(r"(?i)(?:IDR|RP|RM|USD|EUR|AUD|NZD|SGD|MYR|\$|€|£)","",str(value)).strip()
    tokens=re.findall(r"(?<![A-Za-z0-9])[+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*(?![A-Za-z0-9])",s)
    if len(tokens)!=1:return None
    token=re.sub(r"\s*([.,])\s*",r"\1",tokens[0]).replace(" ","")
    sign=-1 if token.startswith("-") else 1;token=token.lstrip("+-")
    if not token or not re.fullmatch(r"\d[\d.,]*",token):return None
    if "," in token and "." in token:
        dec=max(token.rfind(","),token.rfind("."));sep=token[dec]
        chunks=token[:dec].split("." if sep=="," else ",");frac=token[dec+1:]
        if not frac.isdigit() or len(frac) not in (1,2) or not 1<=len(chunks[0])<=3 or any(not x.isdigit() or len(x)!=3 for x in chunks[1:]):return None
        norm="".join(chunks)+"."+frac
    elif "," in token or "." in token:
        sep="," if "," in token else ".";chunks=token.split(sep)
        if any(not x.isdigit() for x in chunks):return None
        if len(chunks)>2:
            if not 1<=len(chunks[0])<=3 or any(len(x)!=3 for x in chunks[1:]):return None
            norm="".join(chunks)
        elif len(chunks)==2 and len(chunks[1])==3 and 1<=len(chunks[0])<=3:norm="".join(chunks)
        elif len(chunks)==2 and len(chunks[1]) in (1,2):norm=chunks[0]+"."+chunks[1]
        else:return None
    else:norm=token
    try:return Decimal(norm)*sign
    except InvalidOperation:return None

_QNUM=re.compile(r"^[\s×xX]*([+-]?\d+(?:(?:\s*[.,]\s*\d+)|(?:\s+\d{3}))*)(.*)$")
def parse_quantity(value):
    if value is None:return None
    s=str(value).strip()
    if re.search(r"\d\s*[xX×]\s*\d",s):return None
    m=_QNUM.fullmatch(s)
    if not m:return None
    number,suffix=m.groups()
    amount=parse_amount(number)
    if amount is None:return None
    unit=suffix.strip().casefold()
    unit=re.sub(r"\s+"," ",unit)
    if unit in {"x","×"}:unit=""
    # A stray x after a count is CORD's multiplier notation, not a unit.
    if unit.startswith(("x ","× ")):unit=unit[2:].strip()
    if unit and not re.fullmatch(r"[a-z%][a-z0-9%/-]*",unit):return None
    return amount,unit

def normalize_menu(value):
    """Normalize dict/list-shaped CORD output into row candidates."""
    if isinstance(value,dict):
        value=value.get("menu",[])
    if isinstance(value,dict):items=[value]
    elif isinstance(value,list):items=value
    else:return []
    out=[]
    for item in items:
        if not isinstance(item,dict):continue
        if not any(item.get(k) not in (None,"") for k in ROW_FIELDS):continue
        out.append(item)
    return out

def donut_rows(donut_json):
    """Extract menu rows from standard object or list-root Donut JSON."""
    if isinstance(donut_json,dict):return normalize_menu(donut_json)
    if isinstance(donut_json,list):
        rows=[]
        for obj in donut_json:
            if isinstance(obj,dict):
                if "menu" in obj:rows.extend(normalize_menu(obj))
                elif any(obj.get(k) not in (None,"") for k in ROW_FIELDS):rows.append(obj)
        return rows
    return []

def _norm_field(field,value):
    if field=="nm":return text_norm(value)
    if field=="cnt":return parse_quantity(value)
    if field in ("price","unitprice","itemsubtotal"):return parse_amount(value)
    return text_norm(value)

def score_rows(gt_rows,pred_rows,support=COMMON_FIELDS):
    """Return TP, FP, FN, eligible-GT count, emitted-candidate count."""
    support=tuple(support)
    targets=[]
    for row in gt_rows if isinstance(gt_rows,list) else []:
        if not isinstance(row,dict):continue
        target={k:row[k] for k in support if row.get(k) not in (None,"")}
        if target:targets.append(target)
    preds=normalize_menu(pred_rows)
    adj=[]
    for pred in preds:
        edges=[]
        for gi,target in enumerate(targets):
            match=True
            for field,truth in target.items():
                if pred.get(field) in (None,""):
                    match=False;break
                pv,tv=_norm_field(field,pred[field]),_norm_field(field,truth)
                if pv is None or pv!=tv:
                    match=False;break
            if match:edges.append(gi)
        adj.append(edges)
    owners={}
    def augment(pi,seen):
        for gi in adj[pi]:
            if gi in seen:continue
            seen.add(gi)
            if gi not in owners or augment(owners[gi],seen):
                owners[gi]=pi;return True
        return False
    tp=sum(augment(pi,set()) for pi in range(len(preds)))
    return tp,len(preds)-tp,len(targets)-tp,len(targets),len(preds)
