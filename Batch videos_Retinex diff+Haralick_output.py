# process_batch_videos.py

import os
import shutil
import numpy as np
import pandas as pd
import cv2, mahotas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib.util
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GT_CSV      = r"C:\Users\Ajay\Desktop\video_gas_frames.csv"
OUTPUT_ROOT = r"C:\Users\Ajay\Desktop\Batch2_Retinex+Haralick_output"
RETINEX_P   = os.path.join(os.path.dirname(__file__), "utils", "Retinex.py")

RADII      = [15, 155]
EPS        = [50.0, 50.0]
WT         = [3.0, 1.0]
PYR_LEVEL  = 2

FEATURE_LABELS = [
    "Energy","Contrast","Correlation","Variance",
    "Homogeneity","SumAverage","SumVariance","SumEntropy",
    "Entropy","DiffVariance","DiffEntropy",
    "InfoMeasureCorr1","InfoMeasureCorr2"
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def parse_ranges(r):
    """Parse gas frame ranges like '0-100,200-300'. Returns list of (start,end) tuples."""
    if pd.isna(r):
        return []
    ranges = []
    for part in str(r).split(','):
        part = part.strip()
        if '-' in part:
            try:
                a, b = map(int, part.split('-'))
                ranges.append((a, b))
            except ValueError:
                pass
    return ranges

def load_retinex(path):
    spec = importlib.util.spec_from_file_location("Retinex", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.guided_filter_multiscale_retinex

# Load Retinex filter once
RTNX = load_retinex(RETINEX_P)

# Clear and recreate output root
if os.path.isdir(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Read ground truth CSV
df_gt = pd.read_csv(GT_CSV)
total_videos = len(df_gt)

for idx, row in df_gt.iterrows():
    video_path = row['video_path']
    ranges     = parse_ranges(row.get('gas_frame_ranges', ''))
    name       = Path(video_path).stem
    out_dir    = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    # Progress print
    pct = (idx+1) / total_videos * 100
    print(f"Processing {idx+1}/{total_videos} ({pct:.1f}%) – {name}")

    # Compute Retinex diffs (no .npy saving)
    data  = np.load(video_path).astype(np.float32)
    diffs = []
    for i in range(1, len(data)):
        p = cv2.normalize(data[i-1], None, 0,255,cv2.NORM_MINMAX)
        c = cv2.normalize(data[i],   None, 0,255,cv2.NORM_MINMAX)
        r1 = RTNX(p, RADII, EPS, WT, pyr_max_level=PYR_LEVEL)
        r2 = RTNX(c, RADII, EPS, WT, pyr_max_level=PYR_LEVEL)
        diffs.append(np.abs(r2 - r1))
    diffs = np.stack(diffs)

    # Compute Haralick + diff summaries
    records = []
    for j, frame in enumerate(diffs):
        m = frame.max()
        img = (frame * 255 / m).astype(np.uint8) if m>0 else np.zeros_like(frame, np.uint8)
        feats = mahotas.features.haralick(img, return_mean=True)
        mean_d, max_d = frame.mean(), frame.max()
        rec = {'frame': j, 'mean_diff': mean_d, 'max_diff': max_d}
        for k, lbl in enumerate(FEATURE_LABELS):
            rec[lbl] = feats[k]
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'haralick.csv'), index=False)

    # STATIC per-video
    fig = go.Figure()
    # diff curves
    fig.add_trace(go.Scatter(x=df.frame, y=df.max_diff,
                             name='Max Diff', line=dict(color='magenta', dash='dash')))
    fig.add_trace(go.Scatter(x=df.frame, y=df.mean_diff,
                             name='Mean Diff', line=dict(color='cyan', dash='dot')))
    # Haralick curves
    for lbl in FEATURE_LABELS:
        fig.add_trace(go.Scatter(x=df.frame, y=df[lbl], name=lbl))
    # gas/no-gas markers
    gas_ix    = [i for i in df.frame if any(a <= i <= b for a,b in ranges)]
    no_gas_ix = [i for i in df.frame if i not in gas_ix]
    fig.add_trace(go.Scatter(
        x=gas_ix,
        y=[df.loc[df.frame==i, 'max_diff'].iat[0] for i in gas_ix],
        mode='markers', marker=dict(color='green', symbol='circle', size=8),
        name='Gas'
    ))
    fig.add_trace(go.Scatter(
        x=no_gas_ix,
        y=[df.loc[df.frame==i, 'max_diff'].iat[0] for i in no_gas_ix],
        mode='markers', marker=dict(color='red', symbol='x', size=8),
        name='No-Gas'
    ))
    # green shading
    for a, b in ranges:
        fig.add_vrect(x0=a, x1=b, fillcolor='LightGreen', opacity=0.3)
    fig.update_layout(
        legend=dict(x=1.02, y=1),
        margin=dict(l=50, r=200, t=50, b=50),
        title=name, xaxis_title='Frame', yaxis_title='Value'
    )
    fig.write_html(os.path.join(out_dir,'static.html'), include_plotlyjs='cdn')

    # DYNAMIC per-video (with moving marker)
    imgs = []
    for j, frame in enumerate(diffs):
        m  = frame.max()
        im = (frame * 255 / m).astype(np.uint8) if m>0 else np.zeros_like(frame, np.uint8)
        cm = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        if j in gas_ix:
            overlay = np.zeros_like(cm)
            overlay[:,:,1] = 255
            cm = cv2.addWeighted(cm, 0.7, overlay, 0.3, 0)
        ds = cv2.resize(cm, (max(1, cm.shape[1]//10), max(1, cm.shape[0]//10)))
        imgs.append(ds)

    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.4,0.6],
                         subplot_titles=('Diff Image','Values vs Frame'))
    fig2.add_trace(go.Image(z=imgs[0]), row=1, col=1)
    # legend entries
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(color='green', symbol='circle', size=10),
                              name='Gas'), row=1, col=2)
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                              marker=dict(color='red', symbol='x', size=10),
                              name='No-Gas'), row=1, col=2)
    # curves
    fig2.add_trace(go.Scatter(x=df.frame, y=df.max_diff, mode='lines',
                              name='Max Diff', line=dict(color='magenta', dash='dash')),
                   row=1, col=2)
    fig2.add_trace(go.Scatter(x=df.frame, y=df.mean_diff, mode='lines',
                              name='Mean Diff', line=dict(color='cyan', dash='dot')),
                   row=1, col=2)
    for lbl in FEATURE_LABELS:
        fig2.add_trace(go.Scatter(x=df.frame, y=df[lbl], mode='lines', name=lbl),
                       row=1, col=2)
    # vertical line
    f0 = df.frame.iloc[0]
    fig2.update_layout(shapes=[dict(type='line', x0=f0, x1=f0,
                                    y0=0, y1=1, xref='x2', yref='paper',
                                    line=dict(color='yellow', width=2))])
    # moving marker
    marker_idx = len(fig2.data)
    fig2.add_trace(go.Scatter(x=[f0], y=[df.max_diff.iloc[0]], mode='markers',
                              marker=dict(color=('green' if 0 in gas_ix else 'red'),
                              size=12), name='Current Frame'),
                   row=1, col=2)
    # slider steps
    steps = []
    for j in range(len(imgs)):
        fr = df.frame.iloc[j]
        yv = df.max_diff.iloc[j]
        clr= 'green' if j in gas_ix else 'red'
        steps.append(dict(label=str(j), method='restyle', args=[
            {'z': [imgs[j]],
             'x': [[fr]],
             'y': [[yv]],
             'marker.color': [[clr]]},
            [0, marker_idx]
        ]))
    fig2.update_layout(sliders=[dict(active=0, pad={'t':50}, steps=steps)],
                       showlegend=True, margin=dict(l=50, r=50, t=50, b=50))
    fig2.write_html(os.path.join(out_dir,'dynamic.html'), include_plotlyjs='cdn')

# BATCH_STATIC: Max/Mean + Prev/Next
def build_batch_static(videos):
    batch = go.Figure()
    for vi, vid in enumerate(videos):
        dfv = pd.read_csv(os.path.join(OUTPUT_ROOT,vid,'haralick.csv'))
        batch.add_trace(go.Scatter(x=dfv.frame, y=dfv.max_diff,
                                   name=f"{vid}-Max Diff", visible=(vi==0)))
        batch.add_trace(go.Scatter(x=dfv.frame, y=dfv.mean_diff,
                                   name=f"{vid}-Mean Diff", visible=(vi==0)))
    buttons=[]
    for direction in ('Prev','Next'):
        vis_sets=[]
        for vi in range(len(videos)):
            idx2 = (vi-1)%len(videos) if direction=='Prev' else (vi+1)%len(videos)
            vis=[False]*len(batch.data)
            for t in (idx2*2, idx2*2+1):
                vis[t]=True
            vis_sets.append(vis)
        buttons.append(dict(label=direction, method='update', args=[{'visible':vis_sets}]))
    batch.update_layout(updatemenus=[dict(type='buttons',buttons=buttons,
                          x=0.5,xanchor='center',y=1.1)],
                        title='Batch Max/Mean', xaxis_title='Frame', yaxis_title='Value')
    batch.write_html(os.path.join(OUTPUT_ROOT,'batch_static.html'), include_plotlyjs='cdn')

# BATCH_DYNAMIC: iframe + Prev/Next wrapper
def build_batch_dynamic(videos):
    html = [
        "<html><body>",
        "<button onclick='prev()'>Prev</button> <button onclick='next()'>Next</button>",
        f"<iframe id='dyn' src='{videos[0]}/dynamic.html' width='100%' height='600px'></iframe>",
        "<script>",
        f"var vids={videos}; var idx=0;",
        "function prev(){idx=(idx-1+vids.length)%vids.length;update();}",
        "function next(){idx=(idx+1)%vids.length;update();}",
        "function update(){document.getElementById('dyn').src=vids[idx]+'/dynamic.html';}",
        "</script></body></html>"
    ]
    with open(os.path.join(OUTPUT_ROOT,'batch_dynamic.html'),'w') as f:
        f.write("\n".join(html))

videos = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT,d))]
build_batch_static(videos)
build_batch_dynamic(videos)
print("Batch processing complete.")
