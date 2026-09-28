# ComfyUI-DesignLayout

Custom node ComfyUI chuyển **việc bố cục chữ sang code (CPU)**: LLM chỉ đưa
**nội dung + ý đồ** (không toạ độ), code lo bố cục, chọn font theo ngôn
ngữ/style, đo chữ bằng font thật, tạo **ảnh hướng dẫn (guide)** để model dựng
nền chừa đúng vùng chữ, và hậu kiểm màu/độ tương phản trên ảnh thật.

Chạy được offline (không GPU) để test bố cục. Không tải model, không mạng —
hợp cho RunningHub serverless.

## Luồng

```
Generate Text (core, Qwen3-4B + system prompt) ─▶ JSON nội dung
   └─▶ DesignJSONParse ──▶ design_spec + background_prompt (đã làm sạch)
          └─▶ DesignLayoutEngine
                 ├─ guide_image ─▶ image1 của TextEncodeQwenImageEditPlus
                 ├─ text_mask
                 ├─ layout_json
                 ├─ background_prompt_augmented ─▶ prompt của TextEncode…
                 └─ debug_preview
   Qwen-Image-Edit 2511 tạo nền (1 lượt sampling)
          └─▶ VAEDecode ─▶ DesignZoneCheck(background, layout_json)
                 ├─ composite_preview ─▶ SaveImage (ẢNH THIẾT KẾ CUỐI, có chữ)
                 ├─ final_json   (không dùng trong workflow)
                 └─ report
```

## Cài đặt

1. Copy thư mục này vào `ComfyUI/custom_nodes/ComfyUI-DesignLayout`.
2. Cài phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
3. **Font đã được đóng gói sẵn** trong `fonts/` (~16MB, 178 font, mỗi font một bản trong thư mục nhóm
   `Decorative/ Other/ San Serif/ Script/ Serif/`). Tham số
   `fonts_dir` mặc định là `"fonts"` → node tự trỏ vào thư mục này bên trong
   package. **Deploy = copy cả thư mục package, không cần cấu hình font gì
   thêm.** `fonts.json` (chỉ mục style + ngôn ngữ) đã build sẵn khớp với `fonts/`.

   > Muốn dùng bộ font khác trên host: đặt `fonts_dir` = đường dẫn tuyệt đối tới
   > thư mục font đó và build lại `fonts.json` (xem "Thêm / đổi font"). Glyph
   > **luôn được kiểm tra lại lúc chạy**, nên font thiếu ký tự tự đổi sang font
   > dự phòng cùng nhóm.

   **Tiếng Việt (và mọi ngôn ngữ có chữ riêng như cyrillic/greek):** khi
   `language` = `vietnamese` chỉ dùng font có `vietnamese` trong `fonts.json`
   (phủ đủ bộ dấu), không bao giờ mượn font Latin dù chuỗi đó chỉ toàn chữ
   không dấu. Nếu LLM ghi sai `language` nhưng chữ có dấu tiếng Việt, Node 1
   tự chuyển sang `vietnamese` (+ cảnh báo).

## Ba node

### DesignJSONParse (Node 1)
- **In:** `llm_output` (STRING).
- **Ra:** `design_spec` (JSON chuẩn hoá), `background_prompt` (đã bỏ cụm từ gây
  ra chữ — **chỉ xoá cụm, không xoá cả câu**), `warnings`.
- Không bao giờ crash: JSON hỏng → sửa dấu phẩy thừa/fence/escape, hỏng hẳn thì
  dùng mặc định + cảnh báo.

### DesignLayoutEngine (Node 2)
- **In:** `design_spec`, `fonts_dir`, `canvas_width/height` (kích thước thiết
  kế cuối, mặc định 1080, tỉ lệ 1:4 … 4:1 — ngoài khoảng này báo lỗi), `bg_width/height` (**0 = tự tính**),
  `seed`, `composition` (auto + danh sách kiểu), `guide_mode`
  (`flat`/`soft_gradient`/`none`), `feather_px`, `zone_padding`,
  `background_prompt` (tuỳ chọn, từ Node 1).
- **Ra:** `guide_image` (IMAGE, cỡ bg) → cắm `image1`; `text_mask` (MASK);
  `layout_json`; `background_prompt_augmented`; `debug_preview` (IMAGE);
  `bg_width`, `bg_height` (INT) → cắm `EmptySD3LatentImage`; `warnings`.
- Cùng `seed` ⇒ cùng kết quả.

### DesignZoneCheck (Node 3)
- **In:** `background` (IMAGE sau VAEDecode), `layout_json`, `fonts_dir`,
  `busy_threshold`, `min_contrast` (mặc định 4.5).
- **Ra:** `final_json`, `composite_preview` (chữ thật lên nền, cỡ canvas —
  đây là ảnh thiết kế cuối), `report`.
- Đo độ rối (Sobel — OpenCV, fallback numpy), trích palette (k-means numpy),
  chọn màu chữ đạt tương phản WCAG; vùng rối → bật shadow/scrim.

## Tham số bố cục chính

| Tham số | Ý nghĩa |
|---|---|
| `composition` | `auto` (chọn theo tỉ lệ khung + `design_type` + seed) hoặc ép 1 kiểu (xem bảng dưới — ép từ widget thì dựng đúng kiểu đó dù không hợp tỉ lệ) |
| `guide_mode` | `flat`: mảng phẳng trong vùng chữ · `soft_gradient`: nền gradient nhẹ · `none`: chỉ mask |
| `feather_px` | Độ mờ viền vùng chữ trên guide (px) |
| `zone_padding` | Nới rộng vùng chữ trên guide (px) |
| `busy_threshold` | Ngưỡng "rối" để bật shadow/scrim (0–1) |
| `min_contrast` | Tương phản WCAG tối thiểu cho màu chữ |

## Kích thước & template theo tỉ lệ khung

`canvas_width/height` quyết định **nhóm tỉ lệ**, mỗi nhóm có bộ template và
trọng số riêng (`designlayout/compositions.py`: `COMPAT`, `WEIGHTS`):

| Nhóm | Tỉ lệ w/h | Ví dụ | Template (auto) |
|---|---|---|---|
| `tall` | < 0.65 | 1080×1920 (story/reels) | `story_top, story_bottom, top_bottom, center_stack, frame_border, badge_stack` |
| `portrait` | 0.65–0.9 | 1080×1350, A4 | `top_headline, top_bottom, bottom_band, center_stack, frame_border, split_diagonal, story_bottom, badge_focus` |
| `square` | 0.9–1.12 | 1080×1080 | `top_headline, center_stack, bottom_band, frame_border, split_diagonal, left_column, right_column, badge_focus` |
| `landscape` | 1.12–2.1 | 1920×1080, 1200×628 | `left_column, right_column, center_wide, lower_third, split_diagonal, frame_border, top_headline, badge_focus` |
| `wide` | 2.1–4 | 1500×500, 1584×396 | `wide_left, wide_right, wide_center, wide_badge` |

Tỉ lệ ngoài 1:4 … 4:1 (vd leaderboard 728×90) **không hỗ trợ**: Layout Engine
báo `ValueError`.

- Template `*badge*` chỉ được chọn tự động khi có chữ `emphasis` (và được cộng
  trọng số khi có).
- `composition_hint` từ LLM không hợp tỉ lệ → đổi sang kiểu tương đương
  (`ALIASES`, vd `left_column` trên khung 9:16 → `story_bottom`) + ghi vào
  `warnings`.
- Lề an toàn, khoảng cách và cỡ chữ co giãn theo cạnh ngắn (cỡ chữ dùng
  `type_unit` = căn(w·h) kẹp trong [cạnh ngắn, 1.25×cạnh ngắn]; banner wide
  được phóng thêm `type_scale`).
- **Nền (latent)**: `bg_width/height = 0` ⇒ cùng tỉ lệ với canvas, ~1MP, bội
  số 16, cạnh dài ≤ 2048 (1080×1080→1024², 1080×1920→768×1360,
  1920×1080→1360×768, 1500×500→1776×592). Zone Check scale nền lên đúng
  canvas rồi ghép chữ.

## Chạy trên RunPod serverless

`workflow_api.json` (sinh bởi `scripts/build_workflow.py`, cùng
`sample_workflow.json`) là workflow dạng API để gửi lên worker. Mỗi request
ghi đè các input:

| Node id | Input | Ý nghĩa |
|---|---|---|
| `2` | `value` | Yêu cầu thiết kế (ngôn ngữ tự nhiên) |
| `31` | `value` | WIDTH (kích thước thiết kế) |
| `32` | `value` | HEIGHT |
| `35` | `sampling_mode.seed` | Seed của LLM |
| `23` | `seed` | Seed sinh nền (nên random mỗi request) |
| `28` | `seed` | Seed chọn bố cục |

Đầu ra: `design_*.png` (**ảnh thiết kế cuối**, node 30) và `background_*.png`
(nền không chữ, node 25 — xoá node này nếu không cần).

**Bộ model (RTX 5090 32GB, mọi model nằm sẵn trên VRAM, ≈ 26.4 GiB weights)** —
đặt trên network volume:

| Node | File | GiB |
|---|---|---|
| 14 UNet (`UnetLoaderGGUF`) | `unet/qwen-image-edit-2511-Q4_K_M.gguf` | 12.3 |
| 15 LoRA | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | 0.8 |
| 18 Text encoder (`CLIPLoader`, type `qwen_image`) | `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | 8.7 |
| 19 VAE | `vae/qwen_image_vae.safetensors` | 0.2 |
| 34 LLM (`CLIPLoader`) | `text_encoders/qwen3.5_2b_bf16.safetensors` | 4.2 |

VRAM còn trống cho activation chỉ ≈ 4.6 GiB: nếu log có `Unloading` /
`loaded partially` thì đổi text encoder sang GGUF Q4_K_M (5.6 GiB, so sánh
bằng `test_workflow.json`).

**LLM lập kế hoạch** dùng node core `TextGenerate` ("Generate Text") với
Qwen3.5-2B (ComfyUI tự nhận loại model từ file; template tắt thinking). System
prompt (node 3) + yêu cầu (node 2) được ghép bằng node core
`StringConcatenate` (node 36) rồi đưa vào `prompt`. Không dùng input
`system_prompt` của `TextGenerate` vì input đó chỉ có trên ComfyUI master sau
22/09/2026 — chưa có trong `runpod/worker-comfyui:5.8.6` (ComfyUI v0.25).
(Không dùng được `qwen_2.5_vl_7b` của Qwen-Image để sinh chữ: ComfyUI nạp nó
không có `lm_head`/stop token.)

Node phụ thuộc ngoài core: ComfyUI-GGUF (`UnetLoaderGGUF`).

## Schema

**Input (LLM)** — xem `prompts/system_prompt.txt`:
```json
{"language":"vietnamese|latin|...","mood":"vintage|elegant|playful|minimal|bold|festive",
 "design_type":"poster|banner|invitation|logo|social",
 "composition_hint":"auto|<tên kiểu>","background_prompt":"...","background_color":"#RRGGBB",
 "texts":[{"role":"headline|subheadline|emphasis|body|detail|note","text":"...","priority":1}]}
```

**final_json** giữ tương thích schema cũ (`text, role, font, size_px, cx, cy,
align, color, letter_spacing, shadow, outline_width, outline_color`) + trường
mới tuỳ chọn (`box, lines, line_height, scrim, shapes, canvas, composition,
seed`). Việc map sang DTO thật của app nằm gọn trong
`designlayout/schema.py::to_app_json` — sửa đúng một chỗ đó khi có định dạng app.

## Thêm kiểu bố cục mới

Trong `designlayout/compositions.py`: viết một hàm `ten_kieu(cw, ch, rng)` trả
về `_base(...)` (định nghĩa 2 dải `primary`/`secondary`, `align`, `valign`,
tuỳ chọn `p_align`/`s_align`, `emphasis_mode="badge"`/`border`,
`type_scale`), rồi thêm vào `BUILDERS`, `COMPAT` (các nhóm tỉ lệ phù hợp) và
`WEIGHTS[<nhóm>]`. Layout engine tự đo chữ và xếp vào dải.

## Thêm / đổi font

1. Bỏ file font vào `fonts/` (thư mục bundled trong package).
2. Build lại chỉ mục:
   ```bash
   python scripts/build_fonts_json.py --root "fonts" --out fonts.json
   ```
3. (Tuỳ chọn) Sửa `group` trong `fonts.json` nếu muốn ép style
   (display/serif/sans/script/decorative). Ngôn ngữ được tính tự động từ glyph.

> Font nằm trong git repo của package. Nếu ngại kích thước, có thể dùng Git LFS
> cho `fonts/**` hoặc gitignore `fonts/` và upload riêng khi deploy.

> **Lưu ý đa ngôn ngữ:** bộ font hiện tại phủ tốt Latin/Cyrillic/Greek nhưng
> rất ít font tiếng Việt (≈5) và gần như không có CJK/Arabic/Thai. Muốn phong
> phú style cho một ngôn ngữ, hãy thêm font phủ ngôn ngữ đó theo từng nhóm
> style rồi build lại `fonts.json`.

## Test & xem thử offline

```bash
# Unit test
python -m pytest -q
# Render 6 mẫu ra samples/ (guide, debug, composite, final)
python scripts/render_samples.py --fonts-dir "/path/to/fonts"
```
