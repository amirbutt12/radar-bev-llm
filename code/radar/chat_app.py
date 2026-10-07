"""Talk to the language models in your browser.

    python chat_app.py            then open http://127.0.0.1:7860

Tab 1  Plain chat : the untouched Qwen2.5-0.5B-Instruct, a normal chat assistant. It knows nothing about the radar.
Tab 2  Ask about a radar frame : pick a nuScenes-mini keyframe, look at its camera picture and radar map, ask a question.
       Three answers side by side: the radar model, the blind model (no radar) and the radar model given a RANDOM other frame's radar.
       These models were only trained on nuScenes-QA style questions, so they answer in one word or number, not in sentences.
"""
import argparse
import io
import random
import threading
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from bev_targets import get_boxes
from pillar_dataset import GRID, ROOT, pillarize
from qa_model import RadarLLM

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--feat-tag", default="mini_mirror")
ap.add_argument("--radar-gpu", type=int, default=0)
ap.add_argument("--chat-gpu", type=int, default=min(1, torch.cuda.device_count() - 1))
args = ap.parse_args()

LLM_DIR = str(ROOT / "weights" / "Qwen2.5-0.5B-Instruct")
FEAT_DIR = ROOT / "cache" / "bev_features" / args.feat_tag
POINTS_DIR = ROOT / "cache" / "radar_points" / "v1.0-mini_sweeps5"
lock = threading.Lock()                                            # one GPU job at a time

print("loading nuScenes mini ...", flush=True)
nusc = NuScenes("v1.0-mini", str(ROOT / "data" / "nuscenes"), verbose=False)
scenes = {}
for s in nusc.scene:
    toks, t = [], s["first_sample_token"]
    while t:
        toks.append(t)
        t = nusc.get("sample", t)["next"]
    scenes[f"{s['name']}: {s['description']}"] = toks

qa = {}                                                            # keyframe token -> [(question, true answer), ...]
for split in ("train", "val"):
    path = ROOT / "cache" / f"nuscenes_qa_{args.feat_tag}_{split}.json"
    if path.exists():
        import json
        for q in json.load(open(path, encoding="utf-8")):
            qa.setdefault(q["sample_token"], []).append((q["question"], q["answer"]))
all_tokens = [t for toks in scenes.values() for t in toks]

print("loading the plain chat model ...", flush=True)
chat_dev = torch.device(f"cuda:{args.chat_gpu}")
chat_tok = AutoTokenizer.from_pretrained(LLM_DIR)
chat_llm = AutoModelForCausalLM.from_pretrained(LLM_DIR, dtype=torch.bfloat16).to(chat_dev).eval()

print("loading the radar and blind models ...", flush=True)
radar_dev = torch.device(f"cuda:{args.radar_gpu}")


def load_qa_model(name, vision):
    m = RadarLLM(LLM_DIR, use_vision=vision).to(radar_dev)
    m.load_state_dict(torch.load(ROOT / "checkpoints" / f"qa_{name}.pt", map_location=radar_dev), strict=False)
    return m.eval()


radar_model, blind_model = load_qa_model("radar", True), load_qa_model("blind", False)
feats = lambda tok: torch.from_numpy(np.load(FEAT_DIR / f"{tok}.npy"))[None]


# ---------------------------------------------------------------- tab 1: plain chat
def chat_fn(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for h in history or []:
        if isinstance(h, dict):
            content = h["content"]
            messages.append({"role": h["role"], "content": content if isinstance(content, str) else str(content)})
        else:
            messages += [{"role": "user", "content": h[0]}, {"role": "assistant", "content": h[1]}]
    messages.append({"role": "user", "content": message if isinstance(message, str) else str(message)})
    text = chat_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = chat_tok(text, return_tensors="pt").to(chat_dev)
    with lock, torch.no_grad():
        out = chat_llm.generate(**ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
    return chat_tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------- tab 2: ask about a radar frame
def frame_token(scene, idx):
    toks = scenes[scene]
    return toks[min(int(idx), len(toks) - 1)]


def camera_image(token):
    sd = nusc.get("sample_data", nusc.get("sample", token)["data"]["CAM_FRONT"])
    img = Image.open(Path(nusc.dataroot) / sd["filename"])
    return img.resize((640, int(640 * img.height / img.width)))


def radar_image(token, show_truth):
    pts = np.load(POINTS_DIR / f"{token}.npy")
    ext = [GRID["y0"] + GRID["ny"] * GRID["cell"], GRID["y0"], GRID["x0"], GRID["x0"] + GRID["nx"] * GRID["cell"]]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(np.minimum(pillarize(pts)[0], 3)[::-1, ::-1], extent=ext, cmap="Greys", vmin=0, vmax=3)
    if show_truth:
        for cls, x, y, w, l, yaw in get_boxes(nusc, token, GRID):
            c, s = np.cos(yaw), np.sin(yaw)
            xy = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]) @ np.array([[c, s], [-s, c]]) + [x, y]
            ax.add_patch(Polygon(np.stack([xy[:, 1], xy[:, 0]], 1), closed=True, fill=False, ec=["limegreen", "magenta"][int(cls)], lw=1.4))
    ax.plot(0, 0, "r^", ms=10)
    ax.set_title(f"radar dots from above (red triangle = our car, up = forward)\n{len(pts)} radar dots", fontsize=9)
    ax.set_xlabel("y (m), left of the car = larger")
    ax.set_ylabel("x (m), forward")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue()))


def show_frame(scene, idx, show_truth):
    tok = frame_token(scene, idx)
    questions = [f"{q}" for q, _ in qa.get(tok, [])]
    return camera_image(tok), radar_image(tok, show_truth), gr.update(choices=questions, value=None), f"keyframe {int(idx)} of this scene | {len(questions)} real nuScenes-QA questions for it"


def random_question(scene, idx):
    items = qa.get(frame_token(scene, idx), [])
    return random.choice(items)[0] if items else ""


def ask(scene, idx, question):
    question = (question or "").strip()
    if not question:
        return "type a question first", "", "", ""
    tok = frame_token(scene, idx)
    other = random.choice([t for t in all_tokens if t not in scenes[scene]])
    with lock:
        a_radar = radar_model.answer([question], feats(tok))[0]
        a_blind = blind_model.answer([question], feats(tok))[0]
        a_wrong = radar_model.answer([question], feats(other))[0]
    truth = next((a for q, a in qa.get(tok, []) if q == question), None)
    return a_radar, a_blind, a_wrong, (truth if truth is not None else "unknown (not one of the dataset's questions for this frame)")


first_scene = next(iter(scenes))
with gr.Blocks(title="radar-bev-llm chat") as demo:
    gr.Markdown("# radar-bev-llm: talk to the models\nSmall models (0.5 billion numbers), trained on a tiny dataset: expect mistakes.")
    with gr.Tab("1. Plain chat"):
        gr.Markdown("The untouched **Qwen2.5-0.5B-Instruct**, like a small ChatGPT. It knows nothing about the radar or the scenes.")
        gr.ChatInterface(chat_fn)
    with gr.Tab("2. Ask about a radar frame"):
        gr.Markdown("Pick a scene and a moment, look at the camera and the radar, then ask a question. **The models only learned nuScenes-QA style questions** "
                    "(e.g. *Are any cars visible?*, *What status is the truck?*, *What number of cars are there?*) and answer in one word or a number. "
                    "The radar-model answer and the wrong-frame answer should differ if the model really reads the radar.")
        with gr.Row():
            scene_dd = gr.Dropdown(list(scenes), value=first_scene, label="Scene")
            idx_sl = gr.Slider(0, 39, value=10, step=1, label="Moment in the scene (keyframe, every 0.5 s)")
            truth_cb = gr.Checkbox(value=True, label="draw the true cars / pedestrians on the radar map")
        info = gr.Markdown()
        with gr.Row():
            cam = gr.Image(label="front camera", type="pil")
            rad = gr.Image(label="radar map", type="pil")
        with gr.Row():
            real_dd = gr.Dropdown([], label="or pick a real question about this frame", allow_custom_value=True)
            rand_btn = gr.Button("random real question")
        q_box = gr.Textbox(label="Your question", placeholder="Are any cars visible?")
        ask_btn = gr.Button("Ask", variant="primary")
        with gr.Row():
            out_radar = gr.Textbox(label="radar model")
            out_blind = gr.Textbox(label="blind model (no radar)")
            out_wrong = gr.Textbox(label="radar model, WRONG frame's radar")
            out_truth = gr.Textbox(label="true answer")
        for trig in (scene_dd.change, idx_sl.release, truth_cb.change, demo.load):
            trig(show_frame, [scene_dd, idx_sl, truth_cb], [cam, rad, real_dd, info])
        real_dd.change(lambda q: q or "", real_dd, q_box)
        rand_btn.click(random_question, [scene_dd, idx_sl], q_box)
        ask_btn.click(ask, [scene_dd, idx_sl, q_box], [out_radar, out_blind, out_wrong, out_truth])
        q_box.submit(ask, [scene_dd, idx_sl, q_box], [out_radar, out_blind, out_wrong, out_truth])

print(f"ready: open http://127.0.0.1:{args.port}", flush=True)
demo.launch(server_name="127.0.0.1", server_port=args.port, show_error=True)
