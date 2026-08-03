"""Generate the portable TF30 screener for 3 August 2026."""
import json
from pathlib import Path
from textwrap import dedent

OUT=Path(__file__).with_name("screen_2026_08_03_tf30.ipynb")
def cell(kind,s): return {"cell_type":kind,"metadata":{},"source":dedent(s).strip().splitlines(True),**({"execution_count":None,"outputs":[]} if kind=="code" else {})}
c=[cell("markdown","""# Kronos TF30 — 3 August 2026

Portable Kaggle/Colab/local screener. Context and anchor end on 31 July. Outputs Top 30 for the first 30-minute bar and the 10-bar session endpoint."""),
cell("code","""%pip install -q --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
%pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow plotly tqdm"""),
cell("code","""from pathlib import Path
import gc, json, os, random, shutil, subprocess, sys
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from IPython.display import display

SEED=42; N_PATHS=5; BATCH_SIZE=32; AS_OF=pd.Timestamp('2026-07-31 23:59:59'); DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type=='cuda':
    cap=torch.cuda.get_device_capability(0); arch=f'sm_{cap[0]}{cap[1]}'; arches=set(torch.cuda.get_arch_list())
    if arch not in arches: raise RuntimeError(f'Restart kernel after torch install; {arch} absent from {sorted(arches)}')
    torch.ones(1,device=DEVICE); torch.cuda.synchronize()
WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content') if Path('/content').exists() else Path.cwd()
REPO=Path.cwd().resolve() if Path.cwd().name=='ISTL' and (Path.cwd()/'.git').exists() else WORK/'ISTL'
if not (REPO/'.git').exists(): subprocess.run(['git','clone','https://github.com/zzeiidann/ISTL.git',str(REPO)],check=True,env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'})
else: subprocess.run(['git','-C',str(REPO),'pull','--ff-only','origin','main'],check=True)
include='Kronos IDX FineTune 30 Minutes/data/idx_kronos_all_30m.parquet,Kronos IDX FineTune 30 Minutes/results/2026-07-30/refit-run-e4/production_model/**'
subprocess.run(['git','-C',str(REPO),'lfs','install','--local'],check=True); subprocess.run(['git','-C',str(REPO),'lfs','pull','--include',include],check=True)
RUNTIME=WORK/'tf30_today_outputs'; RUNTIME.mkdir(parents=True,exist_ok=True)
KDIR=REPO/'Kronos IDX FineTune/Kronos'
if not (KDIR/'model/kronos.py').exists():
    KDIR=RUNTIME/'Kronos-runtime'
    if not (KDIR/'model/kronos.py').exists():
        subprocess.run(['git','clone','https://github.com/shiyu-coder/Kronos.git',str(KDIR)],check=True); subprocess.run(['git','-C',str(KDIR),'checkout','67b630e67f6a18c9e9be918d9b4337c960db1e9a'],check=True)
sys.path.insert(0,str(KDIR)); from model import Kronos,KronosTokenizer,KronosPredictor
DATA=REPO/'Kronos IDX FineTune 30 Minutes/data/idx_kronos_all_30m.parquet'; MODEL=REPO/'Kronos IDX FineTune 30 Minutes/results/2026-07-30/refit-run-e4/production_model'
print({'device':str(DEVICE),'data':str(DATA),'model':str(MODEL),'output':str(RUNTIME)})"""),
cell("code","""F=['open','high','low','close','volume','amount']; raw=pd.read_parquet(DATA); raw['date']=pd.to_datetime(raw.date); raw=raw[raw.date<=AS_OF].sort_values(['ticker','date'])
clocks=sorted(raw.assign(clock=raw.date.dt.strftime('%H:%M')).groupby('clock').size().nlargest(10).index)
schedule=pd.Series([pd.Timestamp('2026-08-03')+pd.Timedelta(hours=int(x[:2]),minutes=int(x[3:])) for x in clocks])
ctx=[]; xt=[]; yt=[]; names=[]; anchors={}
for ticker,g in raw.groupby('ticker'):
    g=g.tail(120)
    if len(g)<120: continue
    ctx.append(g[F].copy()); xt.append(pd.Series(g.date.to_numpy())); yt.append(schedule.copy()); names.append(ticker); anchors[ticker]=float(g.close.iloc[-1])
print('eligible',len(names),'schedule',schedule.tolist())"""),
cell("code","""tok=KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(DEVICE).eval(); model=Kronos.from_pretrained(str(MODEL)).to(DEVICE).eval(); pred=KronosPredictor(model,tok,device=str(DEVICE),max_context=512)
rows=[]
with torch.inference_mode():
  for path in range(N_PATHS):
    torch.manual_seed(SEED+path)
    for start in tqdm(range(0,len(names),BATCH_SIZE),desc=f'path {path+1}'):
      stop=start+BATCH_SIZE; ps=pred.predict_batch(ctx[start:stop],xt[start:stop],yt[start:stop],pred_len=10,T=.8,top_p=.9,top_k=0,sample_count=1,verbose=False)
      for ticker,p in zip(names[start:stop],ps):
        for step in (1,10):
          close=float(p.close.iloc[step-1]); rows.append({'ticker':ticker,'path_id':path,'step':step,'forecast_close':close,'anchor_close':anchors[ticker],'return':close/anchors[ticker]-1})
paths=pd.DataFrame(rows)"""),
cell("code","""def ranking(step):
 d=paths[paths.step.eq(step)].groupby('ticker').agg(anchor_close=('anchor_close','first'),expected_close=('forecast_close','mean'),expected_return=('return','mean'),median_return=('return','median'),probability_up=('return',lambda x:float((x>0).mean())),downside_p10=('return',lambda x:float(np.quantile(x,.1))),dispersion=('return','std')).reset_index().sort_values(['expected_return','probability_up'],ascending=False); d['rank']=range(1,len(d)+1); return d
top30_first=ranking(1).head(30); top30_session=ranking(10).head(30)
print('FIRST BAR 09:00–09:30'); display(top30_first); print('SESSION END'); display(top30_session)
top30_first.to_csv(RUNTIME/'top30_tf30_first_bar_2026_08_03.csv',index=False); top30_session.to_csv(RUNTIME/'top30_tf30_session_2026_08_03.csv',index=False); paths.to_parquet(RUNTIME/'forecast_paths_tf30.parquet',index=False)
(RUNTIME/'run_metadata.json').write_text(json.dumps({'forecast_date':'2026-08-03','context_end':str(AS_OF),'paths':N_PATHS,'eligible':len(names),'warning':'Not investment advice'},indent=2)); print(RUNTIME)""")]
n={"cells":c,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}; OUT.write_text(json.dumps(n,indent=1)); print(OUT)
