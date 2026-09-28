# -*- coding: utf-8 -*-
"""Build a quick test workflow for the Qwen-Image-Edit part of the pipeline.

No ComfyUI-DesignLayout nodes: only core nodes + ComfyUI-GGUF, so the model
set can be checked on a Pod / endpoint before the custom node is deployed.

It tests, in one run:
  1. the planner LLM: system prompt + request (StringConcatenate) ->
     TextGenerate with Qwen3.5-2B -> PreviewAny (JSON text, visible in the UI)
  2. text encoder A/B with the same seed and prompt:
     A = qwen_2.5_vl_7b_fp8_scaled (CLIPLoader)
     B = Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj (CLIPLoaderGGUF)
  3. UNet GGUF + Lightning LoRA on a non-square latent (WIDTH x HEIGHT) with
     a flat guide image built from core nodes (EmptyImage + a lighter plate
     where text would go, like DesignLayoutEngine's guide_mode=flat)

Writes test_workflow.json (UI) and test_workflow_api.json (API format). The
worker returns the SaveImage outputs: test_guide, test_te_fp8, test_te_gguf.
"""
from __future__ import annotations
import os
import json

from build_workflow import (_in, _out, node, to_api, LLM_FILE, LLM_CLIP_TYPE,
                            LLM_JOIN, LORA_FILE, TE_FILE, PROMPT_FILE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "test_workflow.json")
API_DST = os.path.join(ROOT, "test_workflow_api.json")

UNET_FILE = "qwen-image-edit-2511-Q4_K_M.gguf"
TE_FP8_FILE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"

# 9:16 story size the layout engine would pick for a 1080x1920 canvas
WIDTH, HEIGHT = 768, 1360
BG_COLOR = 0x3B2618            # dark brown background_color
PLATE_COLOR = 0x51402F         # +22 per channel, like the flat guide plate
PLATE = {"x": 40, "y": 60, "w": 688, "h": 380}   # top text zone
SEED = 123456

REQUEST = ('Poster khai trương quán cà phê vintage "Mộc Coffee", giảm 30% '
           'tuần đầu, thứ Bảy 12/10, 123 Bạch Đằng Đà Nẵng')
BG_PROMPT = ("A cozy vintage coffee shop interior photograph, a warm wooden "
             "counter with a copper espresso machine, soft golden light, "
             "shallow depth of field. Keep the main subject away from these "
             "areas: the upper center area is a smooth, plain, empty surface "
             "with nothing on it.")

NOTE = f"""TEST NHANH QWEN-IMAGE-EDIT (khong can ComfyUI-DesignLayout)

1. LLM: 'System prompt' + 'Yeu cau' -> Generate Text (Qwen3.5-2B)
   -> xem JSON o 'KET QUA LLM' (chi hien trong UI, worker khong tra text).
2. Text encoder A/B cung seed + prompt:
   test_te_fp8  = qwen_2.5_vl_7b_fp8_scaled (CLIPLoader)
   test_te_gguf = Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj (CLIPLoaderGGUF)
3. UNet GGUF + LoRA Lightning, latent {WIDTH}x{HEIGHT} (doi o WIDTH/HEIGHT),
   guide = nen phang + mang sang o vung chu phia tren (test_guide).

Kiem tra log: 'Using mmproj ...' (GGUF tim thay phan vision), khong co loi
nap model; so sanh 2 anh test_te_* bang mat."""


def build():
    nodes, links = [], []

    def add(n):
        nodes.append(n)
        return n["id"]

    def link(fn, fs, tn, ts, typ):
        links.append([len(links) + 1, fn, fs, tn, ts, typ])

    add(node(1, "Note", "DOC TRUOC", (-900, -420), [], [], [NOTE],
             size=(480, 330)))

    # ---- 1. planner LLM --------------------------------------------------
    with open(PROMPT_FILE, encoding="utf-8") as f:
        sys_prompt = f.read()
    add(node(2, "PrimitiveStringMultiline", "Yeu cau", (-900, -40),
             [_in("value", "STRING", widget=True)], [_out("STRING", "STRING")],
             [REQUEST], size=(400, 120)))
    add(node(3, "PrimitiveStringMultiline", "System prompt", (-900, 120),
             [_in("value", "STRING", widget=True)], [_out("STRING", "STRING")],
             [sys_prompt], size=(400, 200)))
    add(node(4, "StringConcatenate", "System prompt + yeu cau", (-440, 60),
             [_in("string_a", "STRING", widget=True),
              _in("string_b", "STRING", widget=True),
              _in("delimiter", "STRING", widget=True)],
             [_out("STRING", "STRING")], ["", "", LLM_JOIN], size=(300, 130)))
    add(node(5, "CLIPLoader", "LLM (Qwen3.5-2B)", (-440, 240),
             [_in("clip_name", "COMBO", widget=True),
              _in("type", "COMBO", widget=True),
              _in("device", "COMBO", widget=True)],
             [_out("CLIP", "CLIP")], [LLM_FILE, LLM_CLIP_TYPE, "default"],
             size=(300, 110)))
    add(node(6, "TextGenerate", "Generate Text", (-100, 60),
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
             ["", 1024, "on", 0.4, 64, 0.9, 0.05, 1.05, 42, 0.0, False, True],
             size=(380, 400)))
    add(node(7, "PreviewAny", "KET QUA LLM", (320, 60),
             [_in("source", "*")], [], [], size=(420, 300)))
    link(3, 0, 4, 0, "STRING")
    link(2, 0, 4, 1, "STRING")
    link(5, 0, 6, 0, "CLIP")
    link(4, 0, 6, 1, "STRING")
    link(6, 0, 7, 0, "STRING")

    # ---- 3. size + guide image (core nodes) -------------------------------
    int_in = [_in("value", "INT", widget=True)]
    add(node(10, "PrimitiveInt", "WIDTH (latent)", (-900, 560), int_in,
             [_out("INT", "INT")], [WIDTH, "fixed"], size=(260, 82)))
    add(node(11, "PrimitiveInt", "HEIGHT (latent)", (-900, 660),
             [_in("value", "INT", widget=True)], [_out("INT", "INT")],
             [HEIGHT, "fixed"], size=(260, 82)))
    empty_in = [_in("width", "INT", widget=True),
                _in("height", "INT", widget=True),
                _in("batch_size", "INT", widget=True),
                _in("color", "INT", widget=True)]
    add(node(12, "EmptyImage", "Nen guide", (-600, 560), empty_in,
             [_out("IMAGE", "IMAGE")], [WIDTH, HEIGHT, 1, BG_COLOR],
             size=(260, 130)))
    add(node(13, "EmptyImage", "Mang vung chu", (-600, 720),
             [dict(i) for i in empty_in], [_out("IMAGE", "IMAGE")],
             [PLATE["w"], PLATE["h"], 1, PLATE_COLOR], size=(260, 130)))
    add(node(14, "ImageCompositeMasked", "Guide (nen + mang)", (-300, 600),
             [_in("destination", "IMAGE"), _in("source", "IMAGE"),
              _in("x", "INT", widget=True), _in("y", "INT", widget=True),
              _in("resize_source", "BOOLEAN", widget=True),
              _in("mask", "MASK", shape=7)],
             [_out("IMAGE", "IMAGE")],
             [PLATE["x"], PLATE["y"], False], size=(260, 150)))
    add(node(15, "SaveImage", "test_guide", (0, 600),
             [_in("images", "IMAGE"),
              _in("filename_prefix", "STRING", widget=True)],
             [], ["test_guide"], size=(260, 300)))
    link(10, 0, 12, 0, "INT")
    link(11, 0, 12, 1, "INT")
    link(12, 0, 14, 0, "IMAGE")
    link(13, 0, 14, 1, "IMAGE")
    link(14, 0, 15, 0, "IMAGE")

    # ---- shared model chain ------------------------------------------------
    add(node(20, "UnetLoaderGGUF", "UNet GGUF", (320, 460),
             [_in("unet_name", "COMBO", widget=True)], [_out("MODEL", "MODEL")],
             [UNET_FILE], size=(320, 60)))
    add(node(21, "LoraLoaderModelOnly", "LoRA Lightning", (320, 560),
             [_in("model", "MODEL"), _in("lora_name", "COMBO", widget=True),
              _in("strength_model", "FLOAT", widget=True)],
             [_out("MODEL", "MODEL")], [LORA_FILE, 1.0], size=(320, 82)))
    add(node(22, "ModelSamplingAuraFlow", None, (320, 680),
             [_in("model", "MODEL"), _in("shift", "FLOAT", widget=True)],
             [_out("MODEL", "MODEL")], [3.0], size=(320, 60)))
    add(node(23, "VAELoader", None, (320, 780),
             [_in("vae_name", "COMBO", widget=True)], [_out("VAE", "VAE")],
             [VAE_FILE], size=(320, 60)))
    add(node(24, "EmptySD3LatentImage", "Latent WIDTH x HEIGHT", (320, 880),
             [_in("width", "INT", widget=True),
              _in("height", "INT", widget=True),
              _in("batch_size", "INT", widget=True)],
             [_out("LATENT", "LATENT")], [WIDTH, HEIGHT, 1], size=(320, 110)))
    add(node(25, "PrimitiveStringMultiline", "Prompt nen", (-300, 800),
             [_in("value", "STRING", widget=True)], [_out("STRING", "STRING")],
             [BG_PROMPT], size=(400, 150)))
    link(20, 0, 21, 0, "MODEL")
    link(21, 0, 22, 0, "MODEL")
    link(10, 0, 24, 0, "INT")
    link(11, 0, 24, 1, "INT")

    # ---- 2. text encoder A/B -----------------------------------------------
    te_nodes = [
        (30, "CLIPLoader", "A: text encoder fp8",
         [_in("clip_name", "COMBO", widget=True),
          _in("type", "COMBO", widget=True),
          _in("device", "COMBO", widget=True)],
         [TE_FP8_FILE, "qwen_image", "default"], "test_te_fp8"),
        (40, "CLIPLoaderGGUF", "B: text encoder GGUF Q4_K_M + mmproj",
         [_in("clip_name", "COMBO", widget=True),
          _in("type", "COMBO", widget=True)],
         [TE_FILE, "qwen_image"], "test_te_gguf"),
    ]
    for i, (base, ntype, title, ins, widgets, prefix) in enumerate(te_nodes):
        x = 760 + i * 440
        add(node(base, ntype, title, (x, 460), ins, [_out("CLIP", "CLIP")],
                 widgets, size=(400, 110)))
        add(node(base + 1, "TextEncodeQwenImageEditPlus", f"Encode {prefix}",
                 (x, 600),
                 [_in("clip", "CLIP"), _in("vae", "VAE"),
                  _in("image1", "IMAGE"), _in("image2", "IMAGE"),
                  _in("image3", "IMAGE"),
                  _in("prompt", "STRING", widget=True)],
                 [_out("CONDITIONING", "CONDITIONING")], [""],
                 size=(400, 140)))
        add(node(base + 2, "ConditioningZeroOut", "Negative rong", (x, 780),
                 [_in("conditioning", "CONDITIONING")],
                 [_out("CONDITIONING", "CONDITIONING")], [], size=(400, 30)))
        add(node(base + 3, "KSampler", f"KSampler {prefix}", (x, 860),
                 [_in("model", "MODEL"), _in("positive", "CONDITIONING"),
                  _in("negative", "CONDITIONING"),
                  _in("latent_image", "LATENT"),
                  _in("seed", "INT", widget=True),
                  _in("steps", "INT", widget=True),
                  _in("cfg", "FLOAT", widget=True),
                  _in("sampler_name", "COMBO", widget=True),
                  _in("scheduler", "COMBO", widget=True),
                  _in("denoise", "FLOAT", widget=True)],
                 [_out("LATENT", "LATENT")],
                 [SEED, "fixed", 4, 1, "euler", "simple", 1], size=(400, 262)))
        add(node(base + 4, "VAEDecode", None, (x, 1150),
                 [_in("samples", "LATENT"), _in("vae", "VAE")],
                 [_out("IMAGE", "IMAGE")], [], size=(400, 50)))
        add(node(base + 5, "SaveImage", prefix, (x, 1230),
                 [_in("images", "IMAGE"),
                  _in("filename_prefix", "STRING", widget=True)],
                 [], [prefix], size=(400, 400)))
        link(base, 0, base + 1, 0, "CLIP")
        link(23, 0, base + 1, 1, "VAE")
        link(14, 0, base + 1, 2, "IMAGE")
        link(25, 0, base + 1, 5, "STRING")
        link(base + 1, 0, base + 2, 0, "CONDITIONING")
        link(22, 0, base + 3, 0, "MODEL")
        link(base + 1, 0, base + 3, 1, "CONDITIONING")
        link(base + 2, 0, base + 3, 2, "CONDITIONING")
        link(24, 0, base + 3, 3, "LATENT")
        link(base + 3, 0, base + 4, 0, "LATENT")
        link(23, 0, base + 4, 1, "VAE")
        link(base + 4, 0, base + 5, 0, "IMAGE")

    # ---- normalize input.link / output.links + validate ---------------------
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        for o in n.get("outputs", []):
            o["links"] = []
        for i in n.get("inputs", []):
            i["link"] = None
    errs = []
    for lid, fn, fs, tn, ts, typ in links:
        if fn not in by_id or tn not in by_id:
            errs.append(f"link {lid}: missing node")
            continue
        if fs >= len(by_id[fn]["outputs"]) or ts >= len(by_id[tn]["inputs"]):
            errs.append(f"link {lid}: slot out of range")
            continue
        by_id[fn]["outputs"][fs]["links"].append(lid)
        by_id[tn]["inputs"][ts]["link"] = lid
    for n in nodes:
        if n["title"] is None:
            n["title"] = n["type"]
    wf = {"last_node_id": max(by_id), "last_link_id": len(links),
          "nodes": nodes, "links": links, "groups": [], "config": {},
          "extra": {}, "version": 0.4}
    return wf, errs


def main():
    wf, errs = build()
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    api = to_api(wf)
    with open(API_DST, "w", encoding="utf-8") as f:
        json.dump(api, f, ensure_ascii=False, indent=2)
    print("wrote:", DST)
    print("wrote:", API_DST, f"({len(api)} nodes)")
    print("nodes:", len(wf["nodes"]), "links:", len(wf["links"]),
          "errors:", errs or "none")


if __name__ == "__main__":
    main()
