import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('PIPELINE_ROOT', '.')).resolve()
import sys,json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import evalkit as E
import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps
cfg=sys.argv[1]; refs=E.cord_refs('validation')
hit=n=0; sizes=[]
for rid,r in refs.items():
    if r['total'] is None: continue
    n+=1
    im=Image.open(ROOT/f'cord/cord_data/images/clean/{rid}.jpg').convert('RGB'); sizes.append(im.size)
    psm=cfg.split('_')[0]
    if 'gray' in cfg: im=ImageOps.grayscale(im)
    if 'up2' in cfg: im=im.resize((im.width*2,im.height*2),Image.BICUBIC)
    if 'h2000' in cfg and im.height<2000: s=2000/im.height; im=im.resize((round(im.width*s),2000),Image.BICUBIC)
    d=pytesseract.image_to_data(im,config=f"--psm {psm}",output_type=Output.DICT)
    ws=[t.strip() for t in d['text'] if t.strip()]
    cands=ws+[a+' '+b for a,b in zip(ws,ws[1:])]
    t=E.cord_parse_amount(r['total'])
    hit+=any(E.cord_parse_amount(x)==t for x in cands)
import statistics
print(cfg,hit,'/',n,'median size',statistics.median(s[0] for s in sizes),statistics.median(s[1] for s in sizes),flush=True)
