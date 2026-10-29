"""Separate, frozen high-severity CORD robustness extension.

Images are deterministic synthetic variants; inference reruns detector/OCR on
each variant. Existing severity files/results are never overwritten.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

ROOT=Path(__file__).resolve().parent
IMGROOT=ROOT/'cord_data'/'images'
OUTROOT=ROOT/'cord_data'/'results'
CONDITIONS=('blue_cast_high','blue_cast_extreme','fade_extreme','fade_maximum',
            'occlusion_28','occlusion_38','blue_fade_mix','fade_occlusion_mix')
RECIPES={
 'blue_cast_high':{'operation':'blend toward RGB(218,228,255)','blend':0.32},
 'blue_cast_extreme':{'operation':'blend toward RGB(218,228,255)','blend':0.46},
 'fade_extreme':{'operation':'brightness then contrast','brightness':1.10,'contrast':0.30},
 'fade_maximum':{'operation':'brightness then contrast','brightness':1.10,'contrast':0.20},
 'occlusion_28':{'operation':'nested seeded ellipse','additional_area_fraction':0.28,'fill_rgb':[101,82,63]},
 'occlusion_38':{'operation':'nested seeded ellipse','additional_area_fraction':0.38,'fill_rgb':[101,82,63]},
 'blue_fade_mix':{'operation':'blue cast then fade','blend':0.32,'brightness':1.10,'contrast':0.30},
 'fade_occlusion_mix':{'operation':'fade then nested seeded ellipse','brightness':1.10,'contrast':0.20,'additional_area_fraction':0.28,'fill_rgb':[101,82,63]},
}

def apply(im,cond,seed):
    im=im.convert('RGB'); p=RECIPES[cond]
    if cond.startswith('blue_cast'):
        return Image.blend(im,Image.new('RGB',im.size,(218,228,255)),p['blend'])
    if cond=='blue_fade_mix':
        out=Image.blend(im,Image.new('RGB',im.size,(218,228,255)),p['blend'])
        return ImageEnhance.Contrast(ImageEnhance.Brightness(out).enhance(p['brightness'])).enhance(p['contrast'])
    if cond.startswith('fade_') and cond!='fade_occlusion_mix':
        return ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.10)).enhance(p['contrast'])
    if cond in ('occlusion_28','occlusion_38','fade_occlusion_mix'):
        if cond=='fade_occlusion_mix':
            im=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(p['brightness'])).enhance(p['contrast'])
            frac=p['additional_area_fraction']
        else: frac=p['additional_area_fraction']
        # One center per receipt makes higher-area masks nested and paired.
        rng=random.Random(seed+941);w,h=im.size;aspect=w/h
        cx=rng.uniform(.24,.76)*w;cy=rng.uniform(.24,.76)*h
        wf=min(.78,math.sqrt(frac*aspect*4/math.pi));hf=min(.78,math.sqrt(frac*4/(math.pi*aspect)))
        ew=max(1,int(w*wf));eh=max(1,int(h*hf));x0=round(cx-ew/2);y0=round(cy-eh/2)
        out=im.copy();ImageDraw.Draw(out).ellipse((x0,y0,x0+ew,y0+eh),fill=(101,82,63));return out
    raise ValueError(cond)

def export(split,limit=None):
    manifest=json.loads((ROOT/'cord_data'/split/'manifest.json').read_text(encoding='utf-8'))
    if limit is not None:manifest=manifest[:limit]
    outdir=IMGROOT/f'severity_extension_{split}';outdir.mkdir(parents=True,exist_ok=True)
    resultdir=OUTROOT/f'severity_extension_{split}';resultdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for m in manifest:
        rid=m['receipt_id'];seed=m['row_index']
        source=(IMGROOT/'severity_test'/f'{rid}__clean.jpg') if split=='test' else (IMGROOT/'clean'/f'{rid}.jpg')
        im=Image.open(source).convert('RGB');sha=hashlib.sha256(source.read_bytes()).hexdigest()
        for cond in CONDITIONS:
            dest=outdir/f'{rid}__{cond}.jpg';apply(im,cond,seed).save(dest,quality=94,optimize=True)
            rows.append({'receipt_id':rid,'condition':cond,'seed':seed,'source_image':str(source.relative_to(ROOT)),
                'source_sha256':sha,'variant_image':str(dest.relative_to(ROOT)),
                'variant_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'recipe_json':json.dumps(RECIPES[cond],sort_keys=True)})
    with (resultdir/'image_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (resultdir/'config.json').write_text(json.dumps({'split':split,'receipts':len(manifest),'conditions':CONDITIONS,'recipes':RECIPES,
        'paired_mask_policy':'occlusion variants use the same seeded center per receipt; larger masks contain the smaller mask where image boundaries permit',
        'source_policy':'test transforms are generated from the exact clean JPEG inputs used in the prior severity_test suite; clean outputs reused by matching that image hash',
        'synthetic_only':True,'jpeg_quality':94},indent=2),encoding='utf-8')
    print(f'Exported {len(rows)} images ({len(manifest)} receipts x {len(CONDITIONS)} conditions) to {outdir}',flush=True)

def run(split,limit=None):
    from PIL import Image

    manifest=json.loads((ROOT/'cord_data'/split/'manifest.json').read_text(encoding='utf-8'))
    if limit is not None:manifest=manifest[:limit]
    out=OUTROOT/f'severity_extension_{split}';imdir=IMGROOT/f'severity_extension_{split}'
    pending=[]
    for m in manifest:
        for cond in CONDITIONS:
            p=out/f"{m['receipt_id']}__{cond}.json";im=imdir/f"{m['receipt_id']}__{cond}.jpg"
            sha=hashlib.sha256(im.read_bytes()).hexdigest() if im.exists() else None
            try: old=json.loads(p.read_text(encoding='utf-8'));ok=old.get('variant_sha256')==sha and old.get('condition')==cond
            except Exception:ok=False
            if not ok:pending.append((m,cond,p,im,sha))
    if not pending:
        print('All requested predictions already present and hash-valid; no model import.',flush=True);return
    sys.path.insert(0,str(ROOT));import benchmark_cord as b
    print(f'Loading checkpoint pipelines for {len(pending)} missing image cells on {b.DEVICE}',flush=True)
    (lp,lm),(tp,tm),(dp,dm)=b.load_models();started=time.perf_counter();done=0
    for meta,cond,p,impath,sha in pending:
        im=Image.open(impath).convert('RGB');tic=time.perf_counter();words=b.detect(im)
        tw=b.recognize_lines(im,words,tp,tm) if words else []
        lt=b.predict_layout(im,words,lp,lm);ct=b.predict_layout(im,tw,lp,lm)
        dj,ds=b.donut_predict(im,dp,dm)
        rec={'receipt_id':meta['receipt_id'],'condition':cond,'source_split':split,'ground_truth_inference_prohibited':True,
             'variant_sha256':sha,'elapsed_seconds':round(time.perf_counter()-tic,3),'detected_words':len(words),
             'detected_lines':len(set(w['line'] for w in words)),'ocr_text':' '.join(w['text'] for w in words),
             'ocr_lines':b.ocr_line_strings(words),
             'trocr_words':' '.join(w['text'] for w in tw),'ocr_rules_total':b.ocr_rule_total(words),
             'ocr_rows':b.rows_from_words(words),'layout_fields':b.fields_from_tags(lt),'layout_rows':b.tagged_rows(lt),
             'cascade_fields':b.fields_from_tags(ct),'cascade_rows':b.tagged_rows(ct),'donut_json':dj,'donut_sequence':ds}
        p.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');done+=1
        print(json.dumps({'done':done,'requested':len(pending),'receipt':meta['receipt_id'],'condition':cond,'elapsed_s':rec['elapsed_seconds'],'elapsed_total_s':round(time.perf_counter()-started,1)}),flush=True)
    (out/'run_manifest.json').write_text(json.dumps({'split':split,'receipts':len(manifest),'conditions':CONDITIONS,
        'total_matrix_cells':len(manifest)*len(CONDITIONS),'cells_inferred_this_invocation':len(pending),
        'elapsed_seconds':round(time.perf_counter()-started,2),'device':b.DEVICE,
        'pipeline':'Tesseract detection/OCR, TrOCR detector-defined line crops, LayoutLMv3 on detected/provided boxes, Donut full image; rerun per variant',
        'labels_supplied_to_inference':False},indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['export','run','all'],default='all');ap.add_argument('--split',choices=['validation','test'],default='validation');ap.add_argument('--limit',type=int);a=ap.parse_args()
    if a.phase in ('export','all'):export(a.split,a.limit)
    if a.phase in ('run','all'):run(a.split,a.limit)
