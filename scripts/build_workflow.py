# -*- coding: utf-8 -*-
"""Build sample_workflow.json (+ workflow_api.json) from Test (10).json.

Replaces the Regex chain with DesignJSONParse + DesignLayoutEngine and the
AILab QwenVL planner with the core TextGenerate node (Qwen3-4B loaded by
CLIPLoader, system prompt fed to its system_prompt input), wires the
guide image into image1 / VAE into vae / augmented prompt into the Qwen text
encoder, and adds DesignZoneCheck after VAEDecode. WIDTH / HEIGHT primitives
drive the canvas size; the layout engine derives the latent size from them
(bg_width / bg_height -> EmptySD3LatentImage). Re-derives every node's
input.link / output.links from the authoritative links[] array so the graph is
internally consistent.

workflow_api.json is the same graph in ComfyUI API ("prompt") format, the one
a RunPod / serverless worker submits.
"""
from __future__ import annotations
import os
import json

SRC = r"C:\Users\ADMIN\Downloads\Test (10).json"
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "sample_workflow.json")
PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "prompts", "system_prompt.txt")

API_DST = os.path.join(os.path.dirname(DST), "workflow_api.json")

# regex/concat/preview chain, AILab QwenVL planner (4), SaveText json (13)
REMOVE_NODES = {4, 6, 7, 8, 9, 10, 11, 12, 13, 26}

# Model set for an RTX 5090 (32 GB) with every model resident in VRAM
# (~26.4 GiB weights): UNet Q4_K_M 12.3 + LoRA 0.8 + VAE 0.2, Qwen2.5-VL
# fp8_scaled 8.7 (node 18, kept from the source workflow), planner LLM
# Qwen3.5-2B 4.2.
#
# planner LLM: Qwen3.5 is detected from the file itself (CLIPLoader "type" is
# ignored for it) and its template disables thinking when thinking=False. The system prompt is
# joined in front of the request (StringConcatenate) instead of using
# TextGenerate's system_prompt input, which only exists on ComfyUI master
# after 2026-09-22 (not in runpod/worker-comfyui:5.8.6 = ComfyUI v0.25).
LLM_FILE = "qwen3.5_2b_bf16.safetensors"
LLM_CLIP_TYPE = "stable_diffusion"
LLM_JOIN = "\n"  # system_prompt.txt ends with "Request:"

# file names on the RunPod network volume
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# GGUF text encoder, used only by the fp8-vs-GGUF A/B test workflow
# (build_test_workflow.py); ComfyUI-GGUF loads the vision tower from a file
# in the same folder whose name contains "mmproj" and
# "qwen2.5-vl-7b-instruct" (e.g. Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf)
TE_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"

NOTE = """LUONG: yeu cau -> LLM lap ke hoach (Generate Text, Qwen3-4B, JSON noi dung)
  -> 1. Parse JSON -> 2. Layout Engine (bo cuc + guide + prompt nen)
  -> Qwen 2511 tao nen (guide vao image1) -> 6. Zone Check (mau/tuong phan)
  -> 7. THIET KE CUOI: anh da ghep chu (WIDTH x HEIGHT).

1. Go yeu cau vao '1. YEU CAU' (tieng Viet thoai mai).
2. Dat WIDTH / HEIGHT = kich thuoc thiet ke cuoi (1080x1920, 1080x1350,
   1920x1080, 1500x500...; ti le 1:4 .. 4:1). Layout tu chon template hop
   ti le khung (tall / portrait / square / landscape / wide).
3. Kich thuoc nen (latent) tu tinh theo ti le khung, ~1MP, boi so 16.
4. LLM: 'LLM (Qwen3.5-2B)' nap qwen3.5_2b_bf16.safetensors;
   system prompt + yeu cau duoc ghep (Concatenate Text) roi dua vao
   prompt cua Generate Text.
5. Van con chu tren nen? Bo LoRA Lightning, cfg 3-4, 20+ steps, dua
   'text, letters, words, typography, watermark, logo' vao NEGATIVE that."""

CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}


def _in(name, typ, widget=False, shape=None):
    d = {"name": name, "label": name, "type": typ}
    if widget:
        d["widget"] = {"name": name}
    if shape:
        d["shape"] = shape
    return d


def _out(name, typ):
    return {"name": name, "label": name, "type": typ, "links": []}


def node(nid, ntype, title, pos, inputs, outputs, widgets, size=(320, 160)):
    return {
        "id": nid, "type": ntype, "title": title, "pos": list(pos),
        "size": list(size), "flags": {}, "order": nid, "mode": 0,
        "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": ntype}, "widgets_values": widgets,
    }


def main():
    with open(SRC, encoding="utf-8") as f:
        wf = json.load(f)
    with open(PROMPT_FILE, encoding="utf-8") as f:
        sys_prompt = f.read()

    nodes = [n for n in wf["nodes"] if n["id"] not in REMOVE_NODES]
    by_id = {n["id"]: n for n in nodes}

    # refresh the system prompt (node 3) and the read-me note (node 1)
    if 3 in by_id:
        by_id[3]["widgets_values"][0] = sys_prompt
    if 1 in by_id:
        by_id[1]["widgets_values"][0] = NOTE
    if 15 in by_id:
        by_id[15]["widgets_values"][0] = LORA_FILE
    if 22 in by_id:  # latent size now comes from the layout engine
        by_id[22]["title"] = "Nen (tu dong theo WIDTH/HEIGHT)"

    nid = max(n["id"] for n in wf["nodes"]) + 1
    PARSE, LAYOUT, ZONE, SAVEIMG = nid, nid + 1, nid + 2, nid + 3
    WIDTH, HEIGHT, WARN = nid + 4, nid + 5, nid + 6
    LLM_CLIP, LLM, JOIN = nid + 7, nid + 8, nid + 9

    parse = node(PARSE, "DesignJSONParse", "1. Parse JSON", (760, -120),
                 [_in("llm_output", "STRING", widget=True)],
                 [_out("design_spec", "STRING"),
                  _out("background_prompt", "STRING"),
                  _out("warnings", "STRING")], [""], size=(320, 90))
    layout = node(LAYOUT, "DesignLayoutEngine", "2. Layout Engine", (760, 120),
                  [_in("design_spec", "STRING", widget=True),
                   _in("fonts_dir", "STRING", widget=True),
                   _in("canvas_width", "INT", widget=True),
                   _in("canvas_height", "INT", widget=True),
                   _in("bg_width", "INT", widget=True),
                   _in("bg_height", "INT", widget=True),
                   _in("seed", "INT", widget=True),
                   _in("composition", "COMBO", widget=True),
                   _in("guide_mode", "COMBO", widget=True),
                   _in("feather_px", "INT", widget=True),
                   _in("zone_padding", "INT", widget=True),
                   _in("background_prompt", "STRING", widget=True, shape=7)],
                  [_out("guide_image", "IMAGE"), _out("text_mask", "MASK"),
                   _out("layout_json", "STRING"),
                   _out("background_prompt_augmented", "STRING"),
                   _out("debug_preview", "IMAGE"),
                   _out("bg_width", "INT"), _out("bg_height", "INT"),
                   _out("warnings", "STRING")],
                  ["", "fonts", 1080, 1080, 0, 0, 0, "auto", "flat", 24,
                   18, ""], size=(340, 360))
    width = node(WIDTH, "PrimitiveInt", "WIDTH (kich thuoc thiet ke)",
                 (400, 380), [_in("value", "INT", widget=True)],
                 [_out("INT", "INT")], [1080, "fixed"], size=(300, 82))
    height = node(HEIGHT, "PrimitiveInt", "HEIGHT (kich thuoc thiet ke)",
                  (400, 480), [_in("value", "INT", widget=True)],
                  [_out("INT", "INT")], [1080, "fixed"], size=(300, 82))
    warn = node(WARN, "PreviewAny", "Canh bao layout", (1140, 480),
                [_in("source", "*")], [], [], size=(320, 140))
    zone = node(ZONE, "DesignZoneCheck", "6. Zone Check", (2260, 200),
                [_in("background", "IMAGE"),
                 _in("layout_json", "STRING", widget=True),
                 _in("fonts_dir", "STRING", widget=True),
                 _in("busy_threshold", "FLOAT", widget=True),
                 _in("min_contrast", "FLOAT", widget=True)],
                [_out("final_json", "STRING"),
                 _out("composite_preview", "IMAGE"), _out("report", "STRING")],
                ["", "fonts", 0.18, 4.5], size=(320, 160))
    saveimg = node(SAVEIMG, "SaveImage", "7. THIET KE CUOI (anh co chu)",
                   (2620, 200),
                   [_in("images", "IMAGE"),
                    _in("filename_prefix", "STRING", widget=True)],
                   [], ["design"], size=(300, 300))
    llm_clip = node(LLM_CLIP, "CLIPLoader", "LLM (Qwen3.5-2B)", (-480, 560),
                    [_in("clip_name", "COMBO", widget=True),
                     _in("type", "COMBO", widget=True),
                     _in("device", "COMBO", widget=True)],
                    [_out("CLIP", "CLIP")],
                    [LLM_FILE, LLM_CLIP_TYPE, "default"], size=(320, 110))
    join = node(JOIN, "StringConcatenate", "System prompt + yeu cau",
                (-480, 360),
                [_in("string_a", "STRING", widget=True),
                 _in("string_b", "STRING", widget=True),
                 _in("delimiter", "STRING", widget=True)],
                [_out("STRING", "STRING")], ["", "", LLM_JOIN],
                size=(320, 130))
    # widget order = TextGenerate schema order (sampling_mode children follow
    # their combo); inputs present in ComfyUI v0.25+
    llm = node(LLM, "TextGenerate", "2. LLM LAP KE HOACH (Generate Text)",
               (0, 60),
               [_in("clip", "CLIP"),
                _in("prompt", "STRING", widget=True),
                _in("max_length", "INT", widget=True),
                _in("sampling_mode", "COMFY_DYNAMICCOMBO_V3", widget=True),
                _in("sampling_mode.temperature", "FLOAT", widget=True),
                _in("sampling_mode.top_k", "INT", widget=True),
                _in("sampling_mode.top_p", "FLOAT", widget=True),
                _in("sampling_mode.min_p", "FLOAT", widget=True),
                _in("sampling_mode.repetition_penalty", "FLOAT", widget=True),
                _in("sampling_mode.seed", "INT", widget=True),
                _in("sampling_mode.presence_penalty", "FLOAT", widget=True,
                    shape=7),
                _in("thinking", "BOOLEAN", widget=True, shape=7),
                _in("use_default_template", "BOOLEAN", widget=True, shape=7)],
               [_out("generated_text", "STRING")],
               ["", 1024, "on", 0.4, 64, 0.9, 0.05, 1.05, 0, 0.0, False, True],
               size=(400, 400))
    nodes += [parse, layout, zone, saveimg, width, height, warn, llm_clip,
              llm, join]
    by_id = {n["id"]: n for n in nodes}

    # ---- authoritative link list -------------------------------------------
    KEEP = {12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3}
    links = [ln for ln in wf["links"]
             if ln[0] in KEEP and ln[1] not in REMOVE_NODES
             and ln[3] not in REMOVE_NODES]
    lid = max((ln[0] for ln in wf["links"]), default=0)

    def link(fn, fs, tn, ts, typ):
        nonlocal lid
        lid += 1
        links.append([lid, fn, fs, tn, ts, typ])

    link(LLM_CLIP, 0, LLM, 0, "CLIP")         # Qwen3-4B -> Generate Text
    link(3, 0, JOIN, 0, "STRING")             # system prompt -> string_a
    link(2, 0, JOIN, 1, "STRING")             # request -> string_b
    link(JOIN, 0, LLM, 1, "STRING")           # joined -> prompt
    link(LLM, 0, 5, 0, "STRING")              # LLM output -> preview
    link(LLM, 0, PARSE, 0, "STRING")          # LLM output -> parse
    link(PARSE, 0, LAYOUT, 0, "STRING")       # design_spec
    link(PARSE, 1, LAYOUT, 11, "STRING")      # background_prompt (optional)
    link(LAYOUT, 0, 20, 2, "IMAGE")           # guide_image -> image1
    link(19, 0, 20, 1, "VAE")                 # VAE -> vae
    link(LAYOUT, 3, 20, 5, "STRING")          # augmented -> prompt
    link(LAYOUT, 2, ZONE, 1, "STRING")        # layout_json
    link(24, 0, ZONE, 0, "IMAGE")             # decoded bg -> zone
    link(ZONE, 1, SAVEIMG, 0, "IMAGE")        # composite -> SaveImage
    link(WIDTH, 0, LAYOUT, 2, "INT")          # canvas_width
    link(HEIGHT, 0, LAYOUT, 3, "INT")         # canvas_height
    link(LAYOUT, 5, 22, 0, "INT")             # bg_width -> latent width
    link(LAYOUT, 6, 22, 1, "INT")             # bg_height -> latent height
    link(LAYOUT, 7, WARN, 0, "STRING")        # layout warnings

    # ---- normalize node input.link / output.links --------------------------
    for n in nodes:
        for o in n.get("outputs", []):
            o["links"] = []
        for i in n.get("inputs", []):
            i["link"] = None
    for ln in links:
        _id, fn, fs, tn, ts, typ = ln
        if fn in by_id and fs < len(by_id[fn]["outputs"]):
            by_id[fn]["outputs"][fs]["links"].append(_id)
        if tn in by_id and ts < len(by_id[tn]["inputs"]):
            by_id[tn]["inputs"][ts]["link"] = _id

    wf["nodes"] = nodes
    wf["links"] = links
    wf["last_node_id"] = max(n["id"] for n in nodes)
    wf["last_link_id"] = lid

    # ---- validate ----------------------------------------------------------
    errs = []
    ids = {n["id"] for n in nodes}
    for ln in links:
        _id, fn, fs, tn, ts, typ = ln
        if fn not in ids or tn not in ids:
            errs.append(f"link {_id}: missing node")
        elif fs >= len(by_id[fn]["outputs"]) or ts >= len(by_id[tn]["inputs"]):
            errs.append(f"link {_id}: slot out of range")
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    api = to_api(wf)
    with open(API_DST, "w", encoding="utf-8") as f:
        json.dump(api, f, ensure_ascii=False, indent=2)
    print("wrote:", DST)
    print("wrote:", API_DST, f"({len(api)} nodes)")
    print("nodes:", len(nodes), "links:", len(links),
          "errors:", errs or "none")


UI_ONLY = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def to_api(wf):
    """Convert a UI workflow to API format (what /prompt and RunPod accept).

    Muted nodes (mode 2) and UI-only nodes are dropped; bypassed nodes
    (mode 4) are removed and their consumers re-linked to the bypassed
    node's input of the same type. Widget values map to the node's widget
    inputs in order, skipping the control_after_generate value that follows
    INT widgets such as seed. A linked input overrides its widget value.
    """
    by_id = {n["id"]: n for n in wf["nodes"]}
    link_src = {ln[0]: (ln[1], ln[2]) for ln in wf["links"]}

    def source(nid, slot):
        n = by_id[nid]
        while n.get("mode") == 4:  # bypass: pass the matching input through
            typ = n["outputs"][slot]["type"]
            inp = next(i for i in n["inputs"]
                       if i["type"] == typ and i.get("link") is not None)
            nid, slot = link_src[inp["link"]]
            n = by_id[nid]
        return [str(nid), slot]

    api = {}
    for n in wf["nodes"]:
        if n["type"] in UI_ONLY or n.get("mode") in (2, 4):
            continue
        inputs = {}
        vals = list(n.get("widgets_values") or [])
        k = 0
        for i in n.get("inputs", []):
            if i.get("widget"):
                if k < len(vals):
                    inputs[i["name"]] = vals[k]
                k += 1
                if (i["type"] == "INT" and k < len(vals)
                        and vals[k] in CONTROL_VALUES):
                    k += 1
            if i.get("link") is not None:
                inputs[i["name"]] = source(*link_src[i["link"]])
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs,
                             "_meta": {"title": n.get("title") or n["type"]}}
    return api


if __name__ == "__main__":
    main()
