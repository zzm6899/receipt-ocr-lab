"""Ordered synthetic stress matrix on the 100 official CORD validation receipts.

Severity is a simulated image transform; it is not physical receipt damage.
All routing, detector/OCR and model inference runs end-to-end for each image.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parent
base=None
OUT=ROOT/'cord_data'/'results'/'severity_dev';IMG=ROOT/'cord_data'/'images'/'severity_dev'
OUT.mkdir(parents=True,exist_ok=True);IMG.mkdir(parents=True,exist_ok=True)
LEVELS=('mild','moderate','severe');FAMILIES=('fade','blur','occlusion','lighting','crumple')
CONDITIONS=['clean']+[f'{f}_{s}' for f in FAMILIES for s in LEVELS]
PARAMS={
 'fade':{'mild':{'brightness':1.02,'contrast':.82},'moderate':{'brightness':1.06,'contrast':.62},'severe':{'brightness':1.10,'contrast':.42}},
 'blur':{'mild':{'radius':.55},'moderate':{'radius':1.35},'severe':{'radius':2.60}},
 'occlusion':{'mild':{'area_fraction':.025},'moderate':{'area_fraction':.080},'severe':{'area_fraction':.180}},
 'lighting':{'mild':{'shadow_amplitude':.16},'moderate':{'shadow_amplitude':.34},'severe':{'shadow_amplitude':.55}},
 'crumple':{'mild':{'corner_inset':.004,'fold_alpha':28},'moderate':{'corner_inset':.012,'fold_alpha':58},'severe':{'corner_inset':.024,'fold_alpha':98}},
}

def projective(im,seed,strength):
    rng=random.Random(seed);w,h=im.size;unit=min(w,h)*strength
    dst=[(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
    src=[(int(max(0,min(w-1,x+rng.uniform(-1,1)*unit))),int(max(0,min(h-1,y+rng.uniform(-1,1)*unit)))) for x,y in dst]
    A=[];B=[]
    for (x,y),(u,v) in zip(dst,src):
        A.append([x,y,1,0,0,0,-u*x,-u*y]);B.append(u)
        A.append([0,0,0,x,y,1,-v*x,-v*y]);B.append(v)
    coeff=np.linalg.solve(np.array(A,float),np.array(B,float)).tolist()
    return im.transform((w,h),Image.Transform.PERSPECTIVE,coeff,resample=Image.Resampling.BICUBIC)

def crumple(im,seed,severity):
    rng=random.Random(seed); p=PARAMS['crumple'][severity];out=projective(im,seed,p['corner_inset'])
    w,h=out.size;scale=min(w,h);dark=Image.new('RGBA',(w,h),(0,0,0,0));light=Image.new('RGBA',(w,h),(0,0,0,0))
    dd=ImageDraw.Draw(dark);dl=ImageDraw.Draw(light);alpha=int(p['fold_alpha']);width=max(4,int(scale*(.009 if severity=='mild' else .016 if severity=='moderate' else .027)))
    for j in range(2):
        if j==0:a=(int(w*rng.uniform(.18,.40)),0);b=(int(w*rng.uniform(.60,.82)),h-1)
        else:a=(0,int(h*rng.uniform(.22,.44)));b=(w-1,int(h*rng.uniform(.58,.82)))
        c=(int((a[0]+b[0])/2+rng.uniform(-.12,.12)*w),int((a[1]+b[1])/2+rng.uniform(-.10,.10)*h))
        pts=[]
        for k in range(61):
            t=k/60;q=1-t;pts.append((round(q*q*a[0]+2*q*t*c[0]+t*t*b[0]),round(q*q*a[1]+2*q*t*c[1]+t*t*b[1])))
        dd.line(pts,fill=(42,34,27,alpha),width=width);dl.line([(x+2,y) for x,y in pts],fill=(255,255,255,int(alpha*.68)),width=max(2,width//4))
    dark=dark.filter(ImageFilter.GaussianBlur(max(1,int(width*.42))));layer=Image.alpha_composite(dark,light)
    return Image.alpha_composite(out.convert('RGBA'),layer).convert('RGB')

def apply(im,condition,seed):
    im=im.convert('RGB')
    if condition=='clean':return im.copy()
    family,severity=condition.rsplit('_',1);p=PARAMS[family][severity]
    if family=='fade':return ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(p['brightness'])).enhance(p['contrast'])
    if family=='blur':return im.filter(ImageFilter.GaussianBlur(radius=p['radius']))
    if family=='occlusion':
        rng=random.Random(seed+941);w,h=im.size;frac=p['area_fraction'];aspect=w/h
        wf=min(.70,math.sqrt(frac*aspect*4/math.pi));hf=min(.70,math.sqrt(frac*4/(math.pi*aspect)))
        ew=max(1,int(w*wf));eh=max(1,int(h*hf));x=int(rng.uniform(.15,.78)*max(1,w-ew));y=int(rng.uniform(.12,.78)*max(1,h-eh))
        out=im.copy();d=ImageDraw.Draw(out);d.ellipse((x,y,x+ew,y+eh),fill=(101,82,63));return out
    if family=='lighting':
        a=p['shadow_amplitude'];rng=random.Random(seed+1729);w,h=im.size
        cx=rng.uniform(.25,.75)*w;cy=rng.uniform(.25,.75)*h;sx=rng.uniform(.35,.70)*w;sy=rng.uniform(.30,.75)*h
        yy,xx=np.mgrid[0:h,0:w];g=np.exp(-(((xx-cx)**2)/(2*sx*sx)+((yy-cy)**2)/(2*sy*sy)))
        arr=np.asarray(im,dtype=np.float32);arr*=np.clip(1-a*g,.35,1)[...,None];return Image.fromarray(np.uint8(np.clip(arr,0,255)))
    if family=='crumple':return crumple(im,seed,severity)
    raise ValueError(condition)

def export(split='validation',limit=None):
    out=ROOT/'cord_data'/'results'/f'severity_{split}';imgdir=ROOT/'cord_data'/'images'/f'severity_{split}'
    out.mkdir(parents=True,exist_ok=True);imgdir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'cord_data'/split/'manifest.json').read_text(encoding='utf-8'))
    if limit is not None: manifest=manifest[:limit]
    rows=[]
    for m in manifest:
        rid=m['receipt_id'];baseim=Image.open(ROOT/'cord_data'/'images'/'clean'/f'{rid}.jpg').convert('RGB')
        for condition in CONDITIONS:
            im=apply(baseim,condition,m['row_index'])
            path=imgdir/f'{rid}__{condition}.jpg';im.save(path,quality=94,optimize=True)
            rows.append({'receipt_id':rid,'condition':condition,'seed':m['row_index'],
               'source_sha256':hashlib.sha256((ROOT/'cord_data'/'images'/'clean'/f'{rid}.jpg').read_bytes()).hexdigest(),
               'variant_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image':str(path.relative_to(ROOT)),
               'recipe':PARAMS.get(condition.rsplit('_',1)[0],{}) if condition!='clean' else {},
               'parameters':PARAMS.get(condition.rsplit('_',1)[0],{}).get(condition.rsplit('_',1)[1],{}) if condition!='clean' else {}})
    with (out/'severity_image_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'severity_config.json').write_text(json.dumps({'split':f'official CORD {split}','receipts':len(manifest),'conditions':CONDITIONS,
      'parameters':PARAMS,'transform_notes':'one deterministic transform per receipt-condition, seed=row_index; same output on rerun; not physical damage; generated JPEG quality94'},indent=2),encoding='utf-8')
    print(f'Exported {len(rows)} paired synthetic images for {len(manifest)} {split} receipts',flush=True)

def run(split='validation',limit=None):
    out=ROOT/'cord_data'/'results'/f'severity_{split}';imgdir=ROOT/'cord_data'/'images'/f'severity_{split}'
    manifest=json.loads((ROOT/'cord_data'/split/'manifest.json').read_text(encoding='utf-8'))
    if limit is not None: manifest=manifest[:limit]
    global base
    pending=[]
    for m in manifest:
        rid=m['receipt_id']
        for condition in CONDITIONS:
            predout=out/f'{rid}__{condition}.json'; path=imgdir/f'{rid}__{condition}.jpg'
            valid=False
            if predout.exists() and path.exists():
                try:
                    x=json.loads(predout.read_text(encoding='utf-8'))
                    valid=(x.get('condition')==condition and x.get('receipt_id')==rid and x.get('variant_sha256')==hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception:pass
            if not valid:pending.append((m,condition))
    done=0;start=time.perf_counter()
    if pending:
        sys.path.insert(0,str(ROOT));import benchmark_cord as base
        print(f'Loading models on {base.DEVICE}',flush=True);(lp,lm),(tp,tm),(dp,dm)=base.load_models()
    else:
        lp=lm=tp=tm=dp=dm=None
        print(f'All cells already valid; resuming without model imports.',flush=True)
    for m in manifest:
        rid=m['receipt_id']
        for condition in CONDITIONS:
            predout=out/f'{rid}__{condition}.json'
            if predout.exists():
                try:
                    x=json.loads(predout.read_text(encoding='utf-8'))
                    sha=hashlib.sha256((imgdir/f'{rid}__{condition}.jpg').read_bytes()).hexdigest()
                    if x.get('condition')==condition and x.get('receipt_id')==rid and x.get('variant_sha256')==sha:done+=1;continue
                except Exception:pass
            path=imgdir/f'{rid}__{condition}.jpg';im=Image.open(path).convert('RGB');tic=time.perf_counter()
            words=base.detect(im);tw=base.recognize_lines(im,words,tp,tm) if words else []
            lt=base.predict_layout(im,words,lp,lm);ct=base.predict_layout(im,tw,lp,lm);dj,ds=base.donut_predict(im,dp,dm)
            rec={'receipt_id':rid,'condition':condition,'source_split':split,'ground_truth_inference_prohibited':True,
              'variant_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'elapsed_seconds':round(time.perf_counter()-tic,3),
              'detected_words':len(words),'detected_lines':len(set(w['line'] for w in words)),
              'ocr_text':' '.join(w['text'] for w in words),'ocr_lines':base.ocr_line_strings(words),'trocr_words':' '.join(w['text'] for w in tw),
              'ocr_rules_total':base.ocr_rule_total(words),'ocr_rows':base.rows_from_words(words),
              'layout_fields':base.fields_from_tags(lt),'layout_rows':base.tagged_rows(lt),
              'cascade_fields':base.fields_from_tags(ct),'cascade_rows':base.tagged_rows(ct),
              'donut_json':dj,'donut_sequence':ds}
            predout.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');done+=1
            if done%100==0:print(f'{done}/{len(manifest)*len(CONDITIONS)} cells; elapsed {(time.perf_counter()-start)/60:.1f} min',flush=True)
    runmeta={'split':f'official CORD {split}','receipt_count':len(manifest),'conditions':CONDITIONS,
      'image_cells':len(manifest)*len(CONDITIONS),'completed_prediction_cells':done,
      'elapsed_seconds':round(time.perf_counter()-start,2),'device':base.DEVICE if base is not None else 'not_loaded_all_cells_resumed',
      'routing':'end-to-end Tesseract detector/OCR rerun for each saved image; TrOCR receives detector-defined line crops; LayoutLMv3 gets detected boxes; Donut sees only full image',
      'labels':'read only by export/summarize evaluation; never supplied to inference'}
    (out/'severity_run_manifest.json').write_text(json.dumps(runmeta,indent=2),encoding='utf-8');print(json.dumps(runmeta,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['export','run','all'],default='all');ap.add_argument('--split',choices=['validation','test'],default='validation');ap.add_argument('--limit',type=int);a=ap.parse_args()
    if a.phase in ('export','all'):export(a.split,a.limit)
    if a.phase in ('run','all'):run(a.split,a.limit)
