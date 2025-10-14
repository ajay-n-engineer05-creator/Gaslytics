# process_single_video.py
import os, sys, shutil
import numpy as np
import pandas as pd
import cv2, mahotas
import plotly.graph_objects as go
import importlib.util
from pathlib import Path
from plotly.subplots import make_subplots

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH   = r"C:\Users\Ajay\Desktop\npy\0001_Video.npy"
GT_CSV_PATH  = r"C:\Users\Ajay\Desktop\video_gas_frames.csv"
OUTPUT_ROOT  = r"C:\Users\Ajay\Desktop\Retinex+Haralick_output"
RETINEX_PATH = os.path.join(SCRIPT_DIR, "utils", "Retinex.py")

# Retinex parameters
RADII     = [15, 155]
EPSILONS  = [50.0, 50.0]
WEIGHTS   = [3.0, 1.0]
PYR_LEVEL = 2

# Haralick feature labels
FEATURE_LABELS = [
    "Energy", "Contrast", "Correlation", "Variance",
    "Homogeneity", "SumAverage", "SumVariance", "SumEntropy",
    "Entropy", "DiffVariance", "DiffEntropy",
    "InfoMeasureCorr1", "InfoMeasureCorr2"
]

# HELPERS

def parse_ranges(rng_str):
    return [tuple(map(int, part.split("-"))) for part in rng_str.split(",")]


def load_retinex_module(path):
    spec = importlib.util.spec_from_file_location("Retinex", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.guided_filter_multiscale_retinex


def ensure_clean_dir(path):
    if os.path.isdir(path):
        if input(f"{path} exists. Overwrite? (y/n): ").lower() != 'y':
            print("Exiting without changes."); sys.exit(0)
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# ─── MAIN PROCESS ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) Load ground truth
    df_gt = pd.read_csv(GT_CSV_PATH)
    row   = df_gt[df_gt['video_path'] == VIDEO_PATH]
    if row.empty:
        print("No ground-truth entry for this video."); sys.exit(1)
    ranges = parse_ranges(row.iloc[0]['gas_frame_ranges'])

    # 2) Prepare output directory
    video_name = Path(VIDEO_PATH).stem
    out_dir    = os.path.join(OUTPUT_ROOT, video_name)
    ensure_clean_dir(out_dir)

    # 3) Load Retinex function
    RTNX = load_retinex_module(RETINEX_PATH)

    # 4) Compute Retinex diffs
    data  = np.load(VIDEO_PATH).astype(np.float32)
    diffs = []
    total = len(data) - 1
    print("Computing Retinex diffs...")
    for i in range(1, len(data)):
        prev = cv2.normalize(data[i-1], None, 0,255,cv2.NORM_MINMAX)
        curr = cv2.normalize(data[i],   None, 0,255,cv2.NORM_MINMAX)
        r1   = RTNX(prev, RADII, EPSILONS, WEIGHTS, pyr_max_level=PYR_LEVEL)
        r2   = RTNX(curr, RADII, EPSILONS, WEIGHTS, pyr_max_level=PYR_LEVEL)
        diffs.append(np.abs(r2 - r1))
        if i % max(1, total//10) == 0:
            print(f"  {i}/{total} diffs ({(i/total)*100:.1f}%)")
    diffs = np.stack(diffs)
    np.save(os.path.join(out_dir, 'diff_video.npy'), diffs)
    print("Saved diff_video.npy")

    # 5) Compute Haralick + diff summaries
    print("Computing Haralick features and diff summaries...")
    records = []
    for idx, frame in enumerate(diffs):
        # prepare image for Haralick
        m = frame.max()
        img = (frame * 255 / m).astype(np.uint8) if m > 0 else np.zeros_like(frame, dtype=np.uint8)
        feats = mahotas.features.haralick(img, return_mean=True)
        # diff summaries
        mean_diff = frame.mean()
        max_diff  = frame.max()
        # record
        rec = {'frame': idx, 'mean_diff': mean_diff, 'max_diff': max_diff}
        for j, lbl in enumerate(FEATURE_LABELS):
            rec[lbl] = feats[j]
        records.append(rec)
        if idx % max(1, len(diffs)//10) == 0:
            print(f"  {idx}/{len(diffs)} processed ({(idx/len(diffs))*100:.1f}%)")
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'haralick.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved haralick.csv")

    # 6) Prepare gas/no-gas mask indices
    N = len(df)
    gas_mask  = np.zeros(N, dtype=bool)
    for a, b in ranges:
        gas_mask[a: min(b+1, N)] = True
    gas_ix     = np.flatnonzero(gas_mask)
    no_gas_ix  = np.flatnonzero(~gas_mask)

            # 7) STATIC: Plot all features + diff summaries + gas/no-gas markers
    fig = go.Figure()
    # Plot max_diff and mean_diff curves
    fig.add_trace(go.Scatter(
        x=df['frame'], y=df['max_diff'],
        name='Max Diff', line=dict(color='magenta', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df['frame'], y=df['mean_diff'],
        name='Mean Diff', line=dict(color='cyan', dash='dot')
    ))
    # All Haralick feature traces visible by default
    for i, lbl in enumerate(FEATURE_LABELS):
        fig.add_trace(go.Scatter(
            x=df['frame'], y=df[lbl], name=lbl, visible=True
        ))
    # gas/no-gas markers
    fig.add_trace(go.Scatter(
        x=gas_ix, y=df.loc[gas_ix,'max_diff'],
        mode='markers', marker=dict(color='green', symbol='circle', size=8),
        name='Gas'
    ))
    fig.add_trace(go.Scatter(
        x=no_gas_ix, y=df.loc[no_gas_ix,'max_diff'],
        mode='markers', marker=dict(color='red', symbol='x', size=8),
        name='No-Gas'
    ))
    # shaded regions
    for a, b in ranges:
        fig.add_vrect(x0=a, x1=b, fillcolor='LightSalmon', opacity=0.2)
    # Show all traces; legend click toggles visibility
    fig.update_layout(
        legend=dict(x=1.02, y=1, traceorder='normal', font=dict(size=10)),
        margin=dict(l=50, r=200, t=50, b=50),
        title='Diff & Haralick vs Frame', xaxis_title='Frame', yaxis_title='Value'
    )
    fig.write_html(os.path.join(out_dir, 'static.html'), include_plotlyjs='cdn')
    print("Saved static.html")

            # 8) DYNAMIC: image + all curves + current-frame marker + slider
    # Prepare downsampled images with green tint overlay for gas frames
    imgs = []
    for idx, frame in enumerate(diffs):
        m = frame.max()
        im = (frame * 255 / m).astype(np.uint8) if m>0 else np.zeros_like(frame, dtype=np.uint8)
        cm = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        if gas_mask[idx]:
            overlay = np.zeros_like(cm)
            overlay[:, :, 1] = 255
            cm = cv2.addWeighted(cm, 1.0, overlay, 0.3, 0)
        h, w = cm.shape[:2]
        ds = cv2.resize(cm, (max(1, w//10), max(1, h//10)))
        imgs.append(ds)
    # Build subplot
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=('Diff Image', 'Values vs Frame'))
    # Image trace (trace 0)
    fig2.add_trace(go.Image(z=imgs[0]), row=1, col=1)
    # gas/no-gas legend entries
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green', symbol='circle', size=10), name='Gas'), row=1, col=2)
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red',   symbol='x', size=10), name='No-Gas'), row=1, col=2)
    # Plot all curves in col2: max_diff, mean_diff, and Haralick
    fig2.add_trace(go.Scatter(x=df['frame'], y=df['max_diff'], mode='lines', name='Max Diff', line=dict(color='magenta', dash='dash')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=df['frame'], y=df['mean_diff'], mode='lines', name='Mean Diff', line=dict(color='cyan', dash='dot')), row=1, col=2)
    for lbl in FEATURE_LABELS:
        fig2.add_trace(go.Scatter(x=df['frame'], y=df[lbl], mode='lines', name=lbl), row=1, col=2)
    # Add current-frame marker trace
    initial_color = 'green' if gas_mask[0] else 'red'
    marker_idx = len(fig2.data)
    fig2.add_trace(go.Scatter(x=[df['frame'].iloc[0]], y=[df['max_diff'].iloc[0]], mode='markers', marker=dict(color=initial_color, size=12), name='Current Frame'), row=1, col=2)
    # Slider steps: update image (trace 0) and marker (trace marker_idx)
    steps = []
    for i in range(len(imgs)):
        frame_i = df['frame'].iloc[i]
        y_i     = df['max_diff'].iloc[i]
        color_i = 'green' if gas_mask[i] else 'red'
        steps.append(dict(
            label=str(i),
            method='restyle',
            args=[
                {
                    'z':            [imgs[i]],       # update image trace
                    'x':            [[frame_i]],     # update marker x
                    'y':            [[y_i]],         # update marker y
                    'marker.color': [[color_i]]      # update marker color
                },
                [0, marker_idx]
            ]
        ))
    fig2.update_layout(
        sliders=[dict(active=0, pad={'t':50}, steps=steps)],
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig2.write_html(os.path.join(out_dir, 'dynamic.html'), include_plotlyjs='cdn')
    print("Saved dynamic.html")
    print("Single-video processing complete")
