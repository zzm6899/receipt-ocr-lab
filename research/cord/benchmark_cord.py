"""CORD held-out robustness pilot: rerun detector/OCR and models per image variant."""
from __future__ import annotations
import argparse, json, re, time, hashlib, os, sys
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image, ImageDraw, ImageEnhance
import pytesseract
from pytesseract import Output

ROOT=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get("RECEIPT_STORAGE_ROOT", str(Path.home()/"Documents"/"Receipt Storage")))
MODELS=PROJECT/"models"
TESS=PROJECT/r"exports\Receipt-OCR-Portable\runtime\tesseract\tesseract.exe"
os.environ["TESSDATA_PREFIX"]=str(PROJECT/r"exports\Receipt-OCR-Portable\runtime\tesseract\tessdata")
pytesseract.pytesseract.tesseract_cmd=str(TESS)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE=="cuda", "CUDA is required for this pilot"

def corrupt(im: Image.Image, mode: str, seed: int) -> Image.Image:
    im=im.convert("RGB")
    if mode=="clean": return im.copy()
    if mode=="color_cast": return Image.blend(im,Image.new("RGB",im.size,(218,228,255)),.18)
    if mode=="fade": return ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.12)).enhance(.58)
    if mode=="occlusion":
        out=im.copy(); d=ImageDraw.Draw(out); w,h=out.size
        x=int(w*(.78+(.04*(seed%5)/4))); y=int(h*(.35+(.27*(seed%7)/6)))
        d.ellipse((x,y,min(w,x+int(w*.12)),min(h,y+int(h*.11))),fill=(112,91,67))
        return out
    if mode=="rotate180": return im.rotate(180)
    raise ValueError(mode)

def detect(im: Image.Image):
    d=pytesseract.image_to_data(im,config="--psm 6",output_type=Output.DICT)
    words=[]
    for i,t in enumerate(d["text"]):
        t=t.strip()
        if not t: continue
        try: conf=max(0.,min(1.,float(d["conf"][i])/100))
        except: conf=0.
        words.append({"text":t,"left":int(d["left"][i]),"top":int(d["top"][i]),"width":int(d["width"][i]),"height":int(d["height"][i]),"line":(d["page_num"][i],d["block_num"][i],d["par_num"][i],d["line_num"][i]),"conf":conf})
    return words

def ocr_line_strings(words):
    """Preserve the detector's original OCR line segmentation for audit/replay."""
    lines=defaultdict(list)
    for w in words: lines[tuple(w["line"])].append(w)
    ordered=[]
    for _,ws in sorted(lines.items(),key=lambda item:item[0]):
        ordered.append(" ".join(w["text"] for w in sorted(ws,key=lambda x:x["left"])))
    return ordered

def load_models():
    from transformers import AutoProcessor, AutoModelForTokenClassification, TrOCRProcessor, VisionEncoderDecoderModel, DonutProcessor
    lp=AutoProcessor.from_pretrained(str(MODELS/"layoutlmv3-finetuned-cord"),apply_ocr=False)
    lm=AutoModelForTokenClassification.from_pretrained(str(MODELS/"layoutlmv3-finetuned-cord")).to(DEVICE).eval()
    tp=TrOCRProcessor.from_pretrained(str(MODELS/"trocr-base-printed"))
    tm=VisionEncoderDecoderModel.from_pretrained(str(MODELS/"trocr-base-printed")).to(DEVICE).eval()
    dp=DonutProcessor.from_pretrained(str(MODELS/"donut-base-finetuned-cord-v2"))
    dm=VisionEncoderDecoderModel.from_pretrained(str(MODELS/"donut-base-finetuned-cord-v2")).to(DEVICE).eval()
    return (lp,lm),(tp,tm),(dp,dm)

def recognize_lines(im, words, tp, tm):
    """Recognize detector-defined text lines and project tokens into normalized line boxes."""
    lines=defaultdict(list)
    for w in words: lines[w["line"]].append(w)
    ordered=sorted(lines.values(),key=lambda ws:(min(w["top"] for w in ws),min(w["left"] for w in ws)))
    crops=[]; bounds=[]
    for ws in ordered:
        x0=min(w["left"] for w in ws); y0=min(w["top"] for w in ws)
        x1=max(w["left"]+w["width"] for w in ws); y1=max(w["top"]+w["height"] for w in ws)
        bounds.append((x0,y0,x1,y1,ws[0]["line"]))
        crops.append(im.crop((max(0,x0-4),max(0,y0-4),min(im.width,x1+4),min(im.height,y1+4))))
    recognized=[]
    for start in range(0,len(crops),12):
        batch=crops[start:start+12]
        if not batch: continue
        pixels=tp(images=batch,return_tensors="pt").pixel_values.to(DEVICE)
        with torch.inference_mode(): ids=tm.generate(pixels,max_new_tokens=64)
        recognized.extend(tp.batch_decode(ids,skip_special_tokens=True))
    result=[]
    for box,txt in zip(bounds,recognized):
        x0,y0,x1,y1,line=box; toks=txt.strip().split()
        if not toks: continue
        widths=[max(1,len(t)) for t in toks]; denom=sum(widths)
        cursor=x0
        for token,weight in zip(toks,widths):
            nxt=cursor+(x1-x0)*weight/denom
            result.append({"text":token,"left":round(cursor),"top":y0,"width":max(1,round(nxt-cursor)),"height":max(1,y1-y0),"line":line,"conf":0.0})
            cursor=nxt
    return result

def predict_layout(im, words, lp, lm):
    if not words: return []
    width,height=im.size
    boxes=[[max(0,min(1000,int(w["left"]/width*1000))),max(0,min(1000,int(w["top"]/height*1000))),max(0,min(1000,int((w["left"]+w["width"])/width*1000))),max(0,min(1000,int((w["top"]+w["height"])/height*1000)))] for w in words]
    enc=lp(im,[w["text"] for w in words],boxes=boxes,return_tensors="pt",truncation=True)
    word_ids=enc.word_ids(0); enc={k:v.to(DEVICE) for k,v in enc.items()}
    with torch.inference_mode(): logits=lm(**enc).logits[0].softmax(-1)
    labels=logits.argmax(-1).tolist(); conf=logits.max(-1).values.tolist(); id2label=lm.config.id2label
    best={}
    for ti,wi in enumerate(word_ids):
        if wi is None: continue
        if wi not in best or conf[ti]>best[wi][1]: best[wi]=(id2label[labels[ti]],conf[ti])
    return [{**w,"tag":best.get(i,("O",0))[0],"tag_conf":best.get(i,("O",0))[1]} for i,w in enumerate(words)]

def tagged_rows(tagged):
    lines=defaultdict(list)
    for w in tagged: lines[w["line"]].append(w)
    rows=[]
    for line,ws in lines.items():
        r={"nm":[],"cnt":[],"price":[],"unitprice":[],"itemsubtotal":[]}
        for w in ws:
            tag=w["tag"].upper().replace("B-","").replace("I-","")
            if tag.startswith("MENU."):
                f=tag.split(".")[-1].lower()
                if f in r: r[f].append(w["text"])
        row={k:" ".join(v).strip() for k,v in r.items() if v}
        if row: rows.append(row)
    return rows

def fields_from_tags(tagged):
    groups=defaultdict(list)
    for w in tagged:
        t=w["tag"].upper().replace("B-","").replace("I-","")
        if t.startswith("TOTAL.") or t.startswith("SUB_TOTAL."):
            groups[t.split(".",1)[1].lower()].append(w["text"])
    result={}
    for k,vs in groups.items():
        if k=="total_price": result["total"]=" ".join(vs)
    return result

def ocr_rule_total(words):
    lines=defaultdict(list)
    for w in words: lines[w["line"]].append(w["text"])
    money=re.compile(r"(?<![A-Z0-9])(?:[-+]?\d{1,3}(?:[,. ]\d{3})+|[-+]?\d+)(?:[.,]\d{2})?(?![A-Z0-9])",re.I)
    hits=[]
    for toks in lines.values():
        line=" ".join(toks)
        if re.search(r"\b(?:grand\s+)?total\b|amount\s+due|net\s+payable",line,re.I):
            vals=money.findall(line)
            if vals: hits.append(vals[-1])
    return hits[-1] if hits else None

def rows_from_words(words):
    lines=defaultdict(list)
    for w in words: lines[w["line"]].append(w["text"].strip())
    amount=re.compile(r"^\d+(?:[,.]\d{3})+(?:[,.]\d+)?$|^\d+(?:[,.]\d{1,2})$")
    rows=[]
    for toks in lines.values():
        toks=[x for x in toks if x]
        line=" ".join(toks)
        if not line or re.search(r"\b(total|subtotal|change|cash|tax|discount|amount|payment|credit|debit)\b",line,re.I): continue
        indices=[i for i,t in enumerate(toks) if amount.match(t.replace(" ",""))]
        if not indices: continue
        pi=indices[-1]; price=toks[pi]
        name_tokens=toks[:pi]
        count=None
        if len(name_tokens)>1 and re.fullmatch(r"\d{1,2}(?:\.0+)?",name_tokens[0]):
            count=name_tokens[0]; name_tokens=name_tokens[1:]
        name=" ".join(name_tokens).strip(" :*-")
        if name: rows.append({"nm":name,"cnt":count,"price":price})
    return rows

def donut_predict(im, dp, dm):
    pix=dp(im,return_tensors="pt").pixel_values.to(DEVICE)
    ids=dp.tokenizer("<s_cord-v2>",add_special_tokens=False,return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode(): out=dm.generate(pix,decoder_input_ids=ids,max_new_tokens=768)
    seq=dp.batch_decode(out,skip_special_tokens=False)[0]
    return dp.token2json(seq),seq

def donut_total(parsed):
    val=parsed.get("total") if isinstance(parsed,dict) else None
    if isinstance(val,list): val=val[0] if val else None
    if isinstance(val,dict): return val.get("total_price")
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",choices=["validation","test"],default="validation"); ap.add_argument("--limit",type=int,default=5); ap.add_argument("--variants",nargs="+",default=["clean","color_cast","fade","occlusion","rotate180"]); args=ap.parse_args()
    source=ROOT/"cord_data"/args.split/"manifest.json"
    manifest=json.loads(source.read_text(encoding="utf-8"))[:args.limit]
    print(f"Loading models on {DEVICE}: {torch.cuda.get_device_name(0)}",flush=True); models=load_models(); lp,lm=models[0];tp,tm=models[1];dp,dm=models[2]
    outdir=ROOT/"cord_data"/"direct_predictions"/args.split; outdir.mkdir(parents=True,exist_ok=True)
    for meta in manifest:
      for variant in args.variants:
        existing=outdir/f"{meta['receipt_id']}__{variant}.json"
        if existing.exists():
            try:
                old=json.loads(existing.read_text(encoding="utf-8"))
                if old.get("receipt_id")==meta["receipt_id"] and old.get("variant")==variant and old.get("pipeline_revision")=="linecrop_v1":
                    print(json.dumps({"receipt":meta["receipt_id"],"variant":variant,"status":"already_saved"}),flush=True)
                    continue
            except Exception: pass
        path=ROOT/"cord_data"/"images"/variant/f"{meta['receipt_id']}.jpg"; im=Image.open(path).convert("RGB")
        tic=time.perf_counter(); words=detect(im); twords=recognize_lines(im,words,tp,tm) if words else []
        trocr=[w["text"] for w in twords]
        layout=predict_layout(im,words,lp,lm); cascade=predict_layout(im,twords,lp,lm)
        djson,dseq=donut_predict(im,dp,dm)
        rec={"pipeline_revision":"linecrop_v1","receipt_id":meta["receipt_id"],"variant":variant,"source_split":args.split,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"elapsed_seconds":round(time.perf_counter()-tic,3),"detected_words":len(words),"detected_lines":len(set(w["line"] for w in words)),"ocr_text":" ".join(w["text"] for w in words),"ocr_lines":ocr_line_strings(words),"trocr_words":" ".join(trocr),"ocr_rules_total":ocr_rule_total(words),"ocr_rows":rows_from_words(words),"trocr_rows":rows_from_words(twords),"layout_fields":fields_from_tags(layout),"cascade_fields":fields_from_tags(cascade),"layout_rows":tagged_rows(layout),"cascade_rows":tagged_rows(cascade),"donut_json":djson,"donut_sequence":dseq}
        f=outdir/f"{meta['receipt_id']}__{variant}.json"; f.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"receipt":meta["receipt_id"],"variant":variant,"seconds":rec["elapsed_seconds"],"words":len(words),"layout_total":rec["layout_fields"].get("total"),"cascade_total":rec["cascade_fields"].get("total"),"donut_total":donut_total(djson)}),flush=True)

if __name__=="__main__": main()
