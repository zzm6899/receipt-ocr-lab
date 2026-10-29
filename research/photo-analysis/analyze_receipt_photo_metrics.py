import csv,json,hashlib,math,random,statistics
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path
import os
PROJECT_ROOT = os.environ.get('RECEIPT_PROJECT_ROOT', '.')  # folder that holds the dated project directories

os.environ['MPLCONFIGDIR'] = str(Path(PROJECT_ROOT + r'\2026-09-30\task\analysis_handoff_2026-10-07') / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK=Path(PROJECT_ROOT + r'\2026-09-30\task'); OUT=TASK/'analysis_handoff_2026-10-07'
CMP=Path(PROJECT_ROOT + r'\2026-10-05\task\deliverables\qwen3_5_2b_comparison_v1_recovery\receipt_photo_model_results_qwen35_2b_2026-10-06\comparison_all_views.csv')
LEG=TASK/'deliverables/receipy_vscode_workbench/data/labels/legacy_reference_v3_1.json'; NEW=TASK/'deliverables/receipy_vscode_workbench/data/labels/receipy_references_v2.json'
EXT=TASK/'photo_pilot/receipy_extension/scoring_v2'; AMT=TASK/'photo_pilot/combined_cohort_50/field_readability_audit_v1_2'; USR=TASK/'photo_pilot/combined_cohort_50/user_confirmed_field_scoring_v1'
GLM=Path(PROJECT_ROOT + r'\2026-10-04\task-7\evidence\six_model_comparison_v1\deliverables\receipt_photo_model_results_pre_fix_v1_2026-10-05\source_evidence\glm_route_status.json')
QSUM=CMP.parent/'qwen35_2b_addendum_summary.json'; CLASSIC=Path(PROJECT_ROOT + r'\2026-10-04\task-7\evidence\six_model_comparison_v1\attempts\fullcohort_five-route-122-02\classic_four')
QCSV=CMP.parent/'qwen35_2b_known_reference_scores.csv'
QEXEC=CMP.parent/'qwen35_2b_execution_audit.json'
ASSET_MANIFEST=Path(PROJECT_ROOT + r'\2026-10-04\task-7\evidence\six_model_comparison_v1\gpu_smoke_entrypoint_preflight\source_revisions\after_TASK6_fix_before_orchestration_e2e_test\six_model_gpu_entrypoint_manifest_v1.json')
REVIEW_DIR=Path(PROJECT_ROOT + r'\2026-10-01\task\annotation_review\review_packet')
REVIEW_CSV=REVIEW_DIR/'line_item_review.csv'
REVIEW_README=REVIEW_DIR/'README.md'
REVIEW_XLSX=REVIEW_DIR/'receipt_reference_review_v2_1.xlsx'
ROUTES=['Tesseract+rules','LayoutLMv3','TrOCR+LayoutLMv3','Donut','GLM-OCR','Qwen3.5-2B']

def rc(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wc(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dec(v):
 try:return Decimal(str(v).strip()) if str(v).strip() else None
 except InvalidOperation:return None
def wilson(x,n):
 if not n:return(None,None)
 z=1.959963984540054;p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
 return(max(0,c-r),min(1,c+r))
def perc(a,q):
 a=sorted(a);p=(len(a)-1)*q;lo=int(p);hi=math.ceil(p);return a[lo]+(a[hi]-a[lo])*(p-lo) if a else None
def mcnemar(k,n):
 if not n:return 1.0
 masses=[math.comb(n,i)/2**n for i in range(n+1)];return min(1,sum(x for x in masses if x<=masses[k]+1e-15))

OUT.mkdir(parents=True,exist_ok=True)
sourcefiles=[CMP,LEG,NEW,EXT/'score_manifest.json',EXT/'row_receipt_scores.csv',EXT/'row_summary.csv',EXT/'total_decisions.csv',EXT/'total_summary.csv',AMT/'model_condition_amount_retrieval.csv',AMT/'paired_clear_scrunched_amount_deltas.csv',USR/'summary.json',USR/'confirmed_field_presence_metrics.csv',GLM,GLM.parent/'glm_independent_audit_reproduction.json',QSUM,QEXEC,QCSV,ASSET_MANIFEST,REVIEW_CSV,REVIEW_README,REVIEW_XLSX]
sourcefiles=[p for p in sourcefiles if p.exists()]
source_hashes={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sourcefiles}

# Per-receipt inventory counts; the two sources have different scope and annotation certainty.
legacy=json.loads(LEG.read_text(encoding='utf-8')); legacy=legacy if isinstance(legacy,list) else legacy.get('records',[])
ndata=json.loads(NEW.read_text(encoding='utf-8')); newrecs=ndata['records']; rowscore=rc(EXT/'row_receipt_scores.csv')
rowgt={r['receipt_id']:int(r['gt_rows']) for r in rowscore if r['condition']=='clear' and r['route']=='Donut'}
counts=[]
for x in legacy:
 items=x.get('line_items') or []; unk=Counter()
 for it in items:
  for f in ('name','quantity','item_total'):
   if it.get(f) is None or str(it.get(f)).strip().lower() in ('','unknown','unreadable','not printed'):unk[f]+=1
 counts.append({'dataset':'legacy_reference_v3_1','receipt_id':x.get('receipt_id'),'annotated_rows':len(items),'product_rows':len(items),'other_rows':0,'eligible_row_candidates':None,'complete_visible_inventory_candidate':'not stated','reviewer_status':x.get('reviewer_status',''),'coverage_status':x.get('item_coverage_status',''),'unknown_names':unk['name'],'unknown_quantities':unk['quantity'],'unknown_amounts':unk['item_total'],'human_verification':'assistant provisional/unverified'})
for x in newrecs:
 items=x.get('items') or []; products=sum(i.get('row_type')=='product' for i in items); unk=Counter()
 for it in items:
  for f in ('name','quantity','line_total'):
   if it.get(f) is None or str(it.get(f)).strip().lower() in ('','unknown','unreadable','not printed'):unk[f]+=1
 counts.append({'dataset':'receipy_references_v2','receipt_id':x.get('receipt_id'),'annotated_rows':len(items),'product_rows':products,'other_rows':len(items)-products,'eligible_row_candidates':rowgt.get(x.get('receipt_id')),'complete_visible_inventory_candidate':str(bool(x.get('complete_visible_inventory_candidate'))).lower(),'reviewer_status':'assistant-audited','coverage_status':x.get('inventory_scope',''),'unknown_names':unk['name'],'unknown_quantities':unk['quantity'],'unknown_amounts':unk['line_total'],'human_verification':x.get('human_verification','pending')})
wc(OUT/'receipt_item_label_counts.csv',counts)
review_rows=rc(REVIEW_CSV) if REVIEW_CSV.exists() else []
complete_addon=[x for x in newrecs if x.get('complete_visible_inventory_candidate')]
raw_addon_entries=sum(len(x.get('items') or []) for x in complete_addon)
eligible_addon_entries=sum(rowgt.get(x.get('receipt_id'),0) for x in complete_addon)
review_kind=Counter(x.get('Row kind','') for x in review_rows)
reconcile=[
 {'count_basis':'Older scoring snapshot: legacy_reference_v3_1.json','entries':sum(len(x.get('line_items') or []) for x in legacy),'unit':'item rows','eligibility':'unfiltered transcription inventory; 34 receipt records'},
 {'count_basis':'Current review packet v2.1 canonical rows','entries':review_kind.get('Item',0),'unit':'canonical item rows','eligibility':'34 matched instances; assistant-audited, human review pending'},
 {'count_basis':'Current review packet v2.1 separate adjustment','entries':review_kind.get('Non-item adjustment',0),'unit':'non-item adjustment rows','eligibility':'not a product row'},
 {'count_basis':'Review packet Line Items CSV rows','entries':len(review_rows),'unit':'data rows','eligibility':'123 canonical item rows plus separate adjustment'},
 {'count_basis':'Receipy add-on complete-visible ledgers raw entries','entries':raw_addon_entries,'unit':'item/option/adjustment entries','eligibility':'15 complete-visible ledgers; excludes uncertain new_010 and fragment new_014'},
 {'count_basis':'Receipy add-on row-scorer eligible candidates','entries':eligible_addon_entries,'unit':'eligible row candidates','eligibility':'product/option/modifier rows; scorer excludes promotion/discount entries'},
 {'count_basis':'Receipy add-on raw entries excluded by row-score policy','entries':raw_addon_entries-eligible_addon_entries,'unit':'promotion/discount entries','eligibility':'not included as product-row GT candidates'},
 {'count_basis':'Pooled older snapshot + add-on raw entries + separate adjustment','entries':sum(len(x.get('line_items') or []) for x in legacy)+raw_addon_entries+review_kind.get('Non-item adjustment',0),'unit':'review entries','eligibility':'explains 185 count; mixes historical v3_1 with add-on entries'},
 {'count_basis':'Pooled current packet + add-on raw entries + separate adjustment','entries':review_kind.get('Item',0)+raw_addon_entries+review_kind.get('Non-item adjustment',0),'unit':'review entries','eligibility':'188 under newer v2.1 packet; not directly comparable to prior pooled count'}]
wc(OUT/'row_count_reconciliation.csv',reconcile)

# Current 50-pair, six completed routes. The Qwen 4B placeholder is explicitly omitted.
comp=rc(CMP); primary=[x for x in comp if x['route'] in ROUTES and x.get('route_status') in ('ok','generated')]
ms=[]; paired=[]; seed=20261007
for route in ROUTES:
 for cond in ('clear','scrunched'):
  arr=[x for x in primary if x['route']==route and x['condition']==cond]; known=[x for x in arr if x['reference_known']=='1']; tp=sum(x['exact_match']=='1' for x in known);lo,hi=wilson(tp,len(known))
  strict=[x for x in known if x.get('strict_accepted_total','').strip()];stp=sum(dec(x['strict_accepted_total']) is not None and dec(x['strict_accepted_total'])==dec(x['reference_total']) for x in strict)
  ms.append({'route':route,'condition':cond,'view_n':len(arr),'known_reference_n':len(known),'candidate_exact_n':tp,'candidate_accuracy':tp/len(known) if known else None,'wilson95_low':lo,'wilson95_high':hi,'strict_schema_accepted_n':len(strict) if strict else None,'strict_accepted_exact_n':stp if strict else None,'strict_accepted_exact_over_known_n':len(known) if strict else None,'strict_accepted_exact_over_known':stp/len(known) if strict and known else None,'unknown_reference_n':len(arr)-len(known),'errors_or_timeouts':sum(x.get('route_status') not in ('ok','generated') for x in arr),'reference_status':'assistant-audited/unverified'})
 by=defaultdict(dict)
 for x in primary:
  if x['route']==route and x['condition'] in ('clear','scrunched'):by[x['pair_id']][x['condition']]=x
 pairs0=[]
 for pid,v in sorted(by.items()):
  if set(v)=={'clear','scrunched'} and v['clear']['reference_known']=='1' and v['scrunched']['reference_known']=='1':pairs0.append((pid,int(v['clear']['exact_match']=='1'),int(v['scrunched']['exact_match']=='1')))
 b=sum(c==1 and s==1 for _,c,s in pairs0);co=sum(c==1 and s==0 for _,c,s in pairs0);so=sum(c==0 and s==1 for _,c,s in pairs0);n=sum(c==0 and s==0 for _,c,s in pairs0);diff=[s-c for _,c,s in pairs0]
 rng=random.Random(seed);boot=[sum(rng.choice(diff) for _ in diff)/len(diff)*100 for _ in range(10000)] if diff else []
 paired.append({'route':route,'fully_labeled_pairs_n':len(pairs0),'both_correct_n':b,'clear_only_correct_n':co,'scrunched_only_correct_n':so,'neither_correct_n':n,'scrunched_minus_clear_paired_accuracy_pp':sum(diff)/len(diff)*100 if diff else None,'paired_receipt_bootstrap95_low_pp':perc(boot,.025),'paired_receipt_bootstrap95_high_pp':perc(boot,.975),'exact_mcnemar_p_two_sided':mcnemar(so,co+so),'bootstrap_draws':10000,'bootstrap_seed':seed,'reference_status':'assistant-audited/unverified'});seed+=1
order=sorted(range(len(paired)),key=lambda i:paired[i]['exact_mcnemar_p_two_sided'])
prev=0.0
for rank,i in enumerate(order):
 adj=min(1.0,(len(paired)-rank)*paired[i]['exact_mcnemar_p_two_sided'])
 prev=max(prev,adj);paired[i]['holm_adjusted_mcnemar_p_6_routes']=prev
overall=[]
for route in ROUTES:
 arr=[x for x in primary if x['route']==route and x.get('reference_known')=='1']
 tp=sum(x.get('exact_match')=='1' for x in arr);lo,hi=wilson(tp,len(arr))
 strict=[x for x in arr if x.get('strict_accepted_total','').strip()]
 stp=sum(dec(x['strict_accepted_total']) is not None and dec(x['strict_accepted_total'])==dec(x['reference_total']) for x in strict)
 overall.append({'route':route,'known_reference_views_n':len(arr),'candidate_exact_n':tp,'candidate_accuracy':tp/len(arr) if arr else None,'wilson95_low':lo,'wilson95_high':hi,'strict_schema_accepted_n':len(strict) if strict else None,'strict_accepted_exact_n':stp if strict else None,'strict_accepted_exact_accuracy':stp/len(arr) if strict and arr else None,'reference_status':'assistant-audited/unverified'})
wc(OUT/'current_total_accuracy_overall.csv',overall);wc(OUT/'current_total_accuracy_by_model_condition.csv',ms);wc(OUT/'current_paired_clear_scrunched_total_changes.csv',paired)

# Legacy 50 cohort line amount retrieval, EXIF-only pipeline, recall-only multiset endpoint.
amtr=rc(AMT/'model_condition_amount_retrieval.csv'); amtp=rc(AMT/'paired_clear_scrunched_amount_deltas.csv'); line=[]
for x in amtr:
 if x.get('pipeline')!='EXIF-only':continue
 y=next((z for z in amtp if z.get('pipeline')=='EXIF-only' and z.get('model')==x.get('model')),{})
 line.append({'model':x['model'],'condition':x['condition'],'scored_photo_views_n':x['scored_photo_views_n'],'known_amount_targets_n':x['target_amount_values_n'],'exact_amount_values_retrieved':x['matched_amount_values_tp'],'known_amount_values_missed':x['unmatched_known_target_values_fn'],'amount_value_recall':x['known_target_amount_value_recall'],'paired_receipt_clusters_n':y.get('paired_receipt_clusters_n'),'scrunched_minus_clear_recall_pp':y.get('scrunched_minus_clear_known_target_recall_pp'),'paired_bootstrap95_low_pp':y.get('paired_receipt_bootstrap95_low_pp'),'paired_bootstrap95_high_pp':y.get('paired_receipt_bootstrap95_high_pp'),'bootstrap_draws':y.get('bootstrap_draws'),'endpoint_note':'exact multiset amount retrieval only; no name/amount association or full row correctness; assistant-visual labels'})
wc(OUT/'legacy_line_amount_retrieval_exif_only.csv',line)
user=[x for x in rc(USR/'confirmed_field_presence_metrics.csv') if x.get('pipeline')=='EXIF-only baseline'];wc(OUT/'user_confirmed_field_subset.csv',user)

# Separate 16-pair Receipy evaluation. Row PRF1 uses 15 ledgers (fragment excluded); aggregate by item-count strata.
rs=rc(EXT/'row_summary.csv'); total=rc(EXT/'total_summary.csv'); byrec={(x['receipt_id'],x['condition'],x['route']):x for x in rowscore}; ids=sorted(rowgt)
def binof(n):return '1-3 eligible row candidates' if n<=3 else ('4-5 eligible row candidates' if n<=5 else '6+ eligible row candidates')
strata=defaultdict(list)
for rid in ids:strata[binof(rowgt[rid])].append(rid)
strout=[]
for route in ('Tesseract+rules','LayoutLMv3','TrOCR+LayoutLMv3','Donut'):
 for cond in ('clear','scrunched'):
  for binname,rids in sorted(strata.items()):
   a=[byrec[(rid,cond,route)] for rid in rids];tp=sum(int(x['tp']) for x in a);fp=sum(int(x['fp']) for x in a);fn=sum(int(x['fn']) for x in a);pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;f1=2*pr*re/(pr+re) if pr+re else 0
   strout.append({'route':route,'condition':cond,'eligible_row_count_bin':binname,'receipts_n':len(rids),'eligible_gt_row_candidates':sum(int(x['gt_rows']) for x in a),'predicted_rows':sum(int(x['pred_rows']) for x in a),'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':re,'f1':f1,'all_rows_correct_receipts':sum(int(x['tp'])==int(x['gt_rows']) and int(x['fp'])==0 and int(x['fn'])==0 for x in a),'note':'posthoc small stratum; assistant-audited; scorer retains product/option/modifier rows and excludes adjustments/promotions'})
wc(OUT/'receipy_item_count_stratified_row_scores.csv',strout);wc(OUT/'receipy_total_accuracy_by_model_condition.csv',total);wc(OUT/'receipy_row_accuracy_by_model_condition.csv',rs)

# Runtime: only claims supported by per-view telemetry. Classic metric is combined four-route pipeline latency.
runtime=[]
cls=[]
if CLASSIC.exists():
 classic_keys={(x.get('receipt_id'),x.get('condition')) for x in primary if x.get('route') in ('Tesseract+rules','LayoutLMv3','TrOCR+LayoutLMv3','Donut')}
 for f in CLASSIC.glob('*.json'):
  try:
   d=json.loads(f.read_text(encoding='utf-8'))
   matched=(d.get('receipt_id'),d.get('condition')) in classic_keys
   if matched and d.get('elapsed_seconds') is not None and d.get('status')=='complete':cls.append(float(d['elapsed_seconds']))
  except Exception:pass
if cls:runtime.append({'route':'classic four combined sequential pipeline','primary_views_n':len(cls),'mean_seconds_per_view':statistics.mean(cls),'median_seconds_per_view':statistics.median(cls),'p95_seconds_per_view':perc(cls,.95),'model_load_seconds':None,'note':'combined Tesseract/Layout/TrOCR/Donut result latency; do not attribute to individual classic model'})
if GLM.exists():
 g=json.loads(GLM.read_text(encoding='utf-8')); t={x['image_id']:float(x['elapsed_seconds']) for x in g.get('completed',[]) if x.get('elapsed_seconds') is not None};ids0={x['image_id'] for x in primary};vals=[t[k] for k in ids0 if k in t]
 if vals:runtime.append({'route':'GLM-OCR','primary_views_n':len(vals),'mean_seconds_per_view':statistics.mean(vals),'median_seconds_per_view':statistics.median(vals),'p95_seconds_per_view':perc(vals,.95),'model_load_seconds':g.get('load',{}).get('load_seconds'),'note':'route elapsed_seconds, primary views only'})
if QSUM.exists():
 q=json.loads(QSUM.read_text(encoding='utf-8'));runtime.append({'route':'Qwen3.5-2B','primary_views_n':q.get('primary_views'),'mean_seconds_per_view':q.get('mean_total_latency_seconds'),'median_seconds_per_view':q.get('median_total_latency_seconds'),'p95_seconds_per_view':q.get('p95_total_latency_seconds'),'model_load_seconds':2.477,'note':'route summary; load reported separately'})
wc(OUT/'runtime_by_route_summary.csv',runtime)

# Per-image timing table from stored route logs. Classic timings are combined, not per model.
classic_by_key={}
if CLASSIC.exists():
 for f in CLASSIC.glob('*.json'):
  try:
   d=json.loads(f.read_text(encoding='utf-8'))
   if d.get('status')=='complete' and d.get('elapsed_seconds') is not None:
    classic_by_key[(d.get('receipt_id'),d.get('condition'))]=float(d['elapsed_seconds'])
  except (OSError,ValueError,json.JSONDecodeError):
   continue
glm_by_id={}
if GLM.exists():
 g=json.loads(GLM.read_text(encoding='utf-8'))
 glm_by_id={x['image_id']:float(x['elapsed_seconds']) for x in g.get('completed',[]) if x.get('elapsed_seconds') is not None}
qwen_by_id={}
if QCSV.exists():
 for x in rc(QCSV):
  try:qwen_by_id[x['image_id']]={'total':float(x['elapsed_total_seconds']),'generation':float(x['elapsed_generation_seconds'])}
  except (KeyError,ValueError):pass
view_meta={}
for x in primary:
 view_meta.setdefault(x['image_id'],{'image_id':x['image_id'],'receipt_id':x['receipt_id'],'condition':x['condition'],'pair_id':x['pair_id']})
per_image=[]
for image_id,m in sorted(view_meta.items()):
 q=qwen_by_id.get(image_id,{})
 per_image.append({**m,'classic_four_combined_elapsed_seconds':classic_by_key.get((m['receipt_id'],m['condition'])),
  'glm_ocr_elapsed_seconds':glm_by_id.get(image_id),'qwen35_2b_total_elapsed_seconds':q.get('total'),
  'qwen35_2b_generation_elapsed_seconds':q.get('generation'),
  'classic_timing_scope':'combined four-route pipeline; not individually attributable'})
wc(OUT/'per_image_timing.csv',per_image)

# Provenance summary: local asset hashes are identified as hashes, not Hub revisions.
prov=[]
if ASSET_MANIFEST.exists():
 m=json.loads(ASSET_MANIFEST.read_text(encoding='utf-8')); files=m.get('model_files',[]); lin=m.get('model_lineage',{})
 wanted=[('Tesseract+rules','tesseract.exe'),('LayoutLMv3','layoutlmv3-finetuned-cord\\model.safetensors'),('TrOCR recognizer','trocr-base-printed\\model.safetensors'),('Donut','donut-base-finetuned-cord-v2\\pytorch_model.bin')]
 for route,needle in wanted:
  obj=next((z for z in files if z.get('path','').lower().endswith(needle.lower())),{})
  if obj:
   checkpoint_dir=needle.split('\\')[0]
   lineage=lin.get(checkpoint_dir,{})
   prov.append({'route':route,'checkpoint_or_asset':obj.get('path'),'revision':'not recorded for local checkpoint','asset_sha256':obj.get('sha256'),'bytes':obj.get('bytes'),'exposure_or_provenance':lineage.get('training_provenance','Tesseract 5.2.0 plus local extraction rules'),'evidence_basis':'local asset manifest hash; not independently tied to each inference sidecar'})
if GLM.exists():
 g=json.loads(GLM.read_text(encoding='utf-8'));mod=g.get('load',{}).get('model',{})
 prov.append({'route':'GLM-OCR','checkpoint_or_asset':mod.get('repo_id'),'revision':mod.get('revision'),'asset_sha256':g.get('load',{}).get('model_manifest_sha256'),'bytes':g.get('load',{}).get('parameter_bytes'),'exposure_or_provenance':'direct image model; no project fine-tuning in this run','evidence_basis':'route status record; pinned Hub revision'})
if QEXEC.exists():
 q=json.loads(QEXEC.read_text(encoding='utf-8'))
 prov.append({'route':'Qwen3.5-2B','checkpoint_or_asset':q.get('model_repo_id'),'revision':q.get('model_revision'),'asset_sha256':None,'bytes':None,'exposure_or_provenance':'direct image model; no project fine-tuning in this run','evidence_basis':'execution audit; inference label-blind; BF16/nonquantized stated in route summary'})
wc(OUT/'checkpoint_provenance.csv',prov)

# A concise chart with actual current routes only.
labels=['Tesseract','LayoutLMv3','TrOCR + LayoutLMv3','Donut','GLM-OCR','Qwen3.5-2B'];x=list(range(len(ROUTES)));w=.36
clear=[];scr=[];clab=[];slab=[]
for route in ROUTES:
 a=next(x for x in ms if x['route']==route and x['condition']=='clear');b=next(x for x in ms if x['route']==route and x['condition']=='scrunched')
 clear.append(a['candidate_accuracy']*100);scr.append(b['candidate_accuracy']*100);clab.append(f"{a['candidate_exact_n']}/{a['known_reference_n']}");slab.append(f"{b['candidate_exact_n']}/{b['known_reference_n']}")
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained');b1=ax.bar([i-w/2 for i in x],clear,w,label='Clear',color='#4C78A8');b2=ax.bar([i+w/2 for i in x],scr,w,label='Scrunched',color='#F58518')
ax.set_ylim(0,105);ax.set_ylabel('Exact receipt-total match (%)');ax.set_xticks(x,labels,rotation=16,ha='right');ax.set_title('Total accuracy by paired photo condition\n99 known view labels; assistant-audited, unverified references');ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
for i,(u,v) in enumerate(zip(b1,b2)):
 ax.text(u.get_x()+u.get_width()/2,u.get_height()+1,clab[i],ha='center',fontsize=8);ax.text(v.get_x()+v.get_width()/2,v.get_height()+1,slab[i],ha='center',fontsize=8)
fig.savefig(OUT/'total_accuracy_clear_vs_scrunched.png',dpi=180);plt.close(fig)

summary={'main_current_photo_cohort':{'paired_receipt_ids':50,'selected_views':100,'views_each_condition':50,'known_total_refs':99,'unknown_total_refs':1,'extra_unscored_views':22,'completed_routes':ROUTES,'Qwen3.5-4B':'placeholder not run; excluded'},
'legacy_reference_inventory':{'receipts':len(legacy),'line_item_rows':sum(len(x.get('line_items') or []) for x in legacy),'line_item_row_count_distribution':dict(sorted(Counter(len(x.get('line_items') or []) for x in legacy).items())),'human_verified_receipts':0,'partially_user_reviewed_records':sum(x.get('reviewer_status')=='partially_user_reviewed_some_fields_only' for x in legacy)},
'receipy_independent_audit_inventory':{'high_confidence_pairs':16,'selected_views':32,'complete_visible_inventory_ledgers':15,'fragment_excluded_from_row_prf1':1,'eligible_row_candidates_before_field_unknown_filter':58,'eligible_name_targets':56,'eligible_quantity_targets':39,'eligible_line_total_targets':45,'eligible_unit_price_targets':22,'all_human_verified':False},
'user_confirmed_subset':{'receipt_ids':2,'product_fields':3,'user_confirmed_grand_totals':0,'all_four_classic_EXIF_baseline':'0/3 on clear and 0/3 on scrunched; unreviewed output not counted as false positive'},
'cord_context':{'public_release_receipts':1000,'train':800,'validation':100,'test':100,'local_CORD_finetuned_checkpoint_testID_exposure':'not established'},
'fewshot_feasibility_context':{'LayoutLMv3_smokes':'10/25/50 labeled spans per field; seeds13/29/47; 1 epoch; 48/113/204 train receipts; 4.70/10.01/18.21 s loop; oracle GT text/boxes; no validation/test accuracy', 'Donut_smoke':'no completed result; no accuracy claim'},
'paired_stats':'Current main total score: pair-level bootstrap over 49 fully labeled pairs; marginal accuracy uses 50 clear and 49 scrunched known refs. All labels are assistant-audited/unverified.', 'input_files':source_hashes}
(OUT/'analysis_evidence.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'current_totals':ms,'paired_total_changes':paired,'line_amounts':line,'row_count_strata':strout,'runtime':runtime},indent=2,ensure_ascii=False))
