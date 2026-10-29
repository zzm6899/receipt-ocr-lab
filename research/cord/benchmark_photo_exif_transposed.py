"""Rerun the frozen 34 clear/scrunched photo pairs after honoring EXIF orientation.

This is a loader-correction ablation. It reuses the frozen selected filenames and
same model checkpoints/inference functions as the original photo pass, without
consulting receipt labels or reference text during inference. Baseline files are
read only and never overwritten. Outputs stay in a new local-only directory.
"""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get("RECEIPT_STORAGE_ROOT", str(Path.home()/"Documents"/"Receipt Storage")))
DATA=PROJECT/"training_data"
MANIFEST=ROOT/"photo_pilot"/"provisional_scoring_v2"/"photo_scoring_manifest.json"
OUT=ROOT/"photo_pilot"/"exif_transposed_predictions"
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from benchmark_cord import (load_models,detect,recognize_lines,predict_layout,
    fields_from_tags,tagged_rows,donut_predict,ocr_rule_total,rows_from_words)

def find_image(split,name,index):
    candidates=index.get((split,name),[])
    if len(candidates)!=1:
        raise RuntimeError(f"Expected one {split} image named {name!r}, found {len(candidates)}")
    return candidates[0]

def first_nonempty_total(obj):
    nodes=obj if isinstance(obj,list) else [obj]
    for node in nodes:
        if not isinstance(node,dict): continue
        value=node.get("total",node.get("total_price"))
        vals=value if isinstance(value,list) else [value]
        for v in vals:
            if isinstance(v,dict):
                x=v.get("total_price")
                if x not in (None,"") and not isinstance(x,(list,dict)): return x
            elif isinstance(v,(str,int,float)) and str(v).strip() and "total" in node:
                return v
    return None

def donor_rows(obj):
    if isinstance(obj,list):
        # Preserve model-generated structures; normalize only supported menu shape.
        allrows=[]
        for x in obj:
            if isinstance(x,dict):
                m=x.get("menu")
                if isinstance(m,dict): allrows.append(m)
                elif isinstance(m,list): allrows.extend(y for y in m if isinstance(y,dict))
        return allrows if allrows else [x for x in obj if isinstance(x,dict) and any(k in x for k in ("nm","cnt","price"))]
    if isinstance(obj,dict):
        m=obj.get("menu",[])
        if isinstance(m,dict): return [m]
        if isinstance(m,list): return [x for x in m if isinstance(x,dict)]
    return []

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Index only filenames/paths. The inference loop never opens the labels.
    index={}
    for split in ("clear","scrunched"):
        for p in (DATA/split).rglob("*"):
            if p.is_file(): index.setdefault((split,p.name),[]).append(p)
    inputs=[]
    for inst in manifest["instances"]:
        for split,key in (("clear","clear_image"),("scrunched","scrunched_image")):
            path=find_image(split,inst[key],index)
            inputs.append((inst["receipt_id"],split,path))
    if len(inputs)!=68 or len({(rid,cond) for rid,cond,_ in inputs})!=68:
        raise RuntimeError(f"Expected 68 unique paired views, got {len(inputs)}")
    print("Loading the same cached checkpoints used by the original photo baseline",flush=True)
    (lp,lm),(tp,tm),(dp,dm)=load_models()
    t0=time.perf_counter()
    for i,(rid,cond,path) in enumerate(inputs,1):
        dest=OUT/f"{rid}__{cond}.json"
        if dest.exists():
            # Safe resumability requires matching original source bytes.
            prev=json.loads(dest.read_text(encoding="utf-8"))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if prev.get("source_sha256")==digest and prev.get("pipeline_revision")=="exif_transpose_v1":
                print(f"[{i}/68] resume verified {rid}/{cond}",flush=True)
                continue
            raise RuntimeError(f"Existing output provenance mismatch at {dest}")
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as opened:
            stored_size=list(opened.size);exif_orientation=opened.getexif().get(274,1)
            image=ImageOps.exif_transpose(opened).convert("RGB")
        pixel_hash=hashlib.sha256(image.tobytes()).hexdigest()
        words=detect(image)
        trocr=recognize_lines(image,words,tp,tm) if words else []
        layout=predict_layout(image,words,lp,lm)
        cascade=predict_layout(image,trocr,lp,lm)
        donut_json,donut_sequence=donut_predict(image,dp,dm)
        record={"pipeline_revision":"exif_transpose_v1","receipt_id":rid,"condition":cond,
            "filename":path.name,"source_path_class":cond,"source_sha256":source_hash,
            "stored_size":stored_size,"inference_size":list(image.size),"exif_orientation":exif_orientation,
            "exif_transposed":True,"inference_pixel_sha256":pixel_hash,
            "elapsed_seconds":round(time.perf_counter()-t0,3),"detected_words":len(words),
            "detected_lines":len({w["line"] for w in words}),"ocr_text":" ".join(w["text"] for w in words),
            "ocr_total":ocr_rule_total(words),"layout_total":fields_from_tags(layout).get("total"),
            "cascade_total":fields_from_tags(cascade).get("total"),"donut_total":first_nonempty_total(donut_json),
            "ocr_rows":rows_from_words(words),"layout_rows":tagged_rows(layout),"cascade_rows":tagged_rows(cascade),
            "donut_rows":donor_rows(donut_json),"donut_json":donut_json,"donut_sequence":donut_sequence}
        dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"[{i}/68] {rid}/{cond}: EXIF={exif_orientation} {stored_size}->{list(image.size)} words={len(words)} totals={record['ocr_total']!r}/{record['layout_total']!r}/{record['cascade_total']!r}/{record['donut_total']!r}",flush=True)
    print(f"Completed EXIF-transposed inference for {len(inputs)} views in {time.perf_counter()-t0:.1f}s; outputs={OUT}",flush=True)

if __name__=="__main__": main()
