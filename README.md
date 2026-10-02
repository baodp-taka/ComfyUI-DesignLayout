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

### DesignLayoutFromImage (luồng image-first, khuyến nghị)
Chọn bố cục **sau khi đã có ảnh nền**, giống cách designer làm:
1. Sinh nền **tự do** (không guide, không mask) → ảnh tự nhiên.
2. Node chấm điểm mọi template hợp `design_type` + khổ ảnh theo **độ rối
   dưới từng dòng chữ** (50% trung bình + 50% đoạn tệ nhất trên dòng, cộng độ
   lệch sáng tối), chọn chỗ thoáng nhất; trong các ứng viên cách tốt nhất
   ≤ 15% thì seed quyết định. Template có tấm nền bị loại (giữ ảnh tự nhiên).
3. **Xoá phông** quanh từng dòng, mạnh/nhẹ theo độ rối — không dùng scrim xám.
4. Chỉnh màu chữ trên nền đã xử lý rồi ghép chữ.

- **In:** `background` (IMAGE sau VAEDecode), `design_spec`, `fonts_dir`,
  `canvas_width/height`, `seed`, `composition` (`auto` = chấm điểm; tên
  template = ép), `candidates` (0 = tất cả), `defocus` (0 tắt · 1 mặc định ·
  2 mạnh), `min_contrast`, `text_effect` (tuỳ chọn, mặc định `none`).
- **`text_effect`** (`designlayout/text_fx.py`): chữ do code vẽ nên luôn đúng
  chính tả / dấu. `none` = chữ phẳng như cũ; `auto` = chọn kiểu theo mood +
  độ sáng nền + seed; hoặc ép một kiểu:
  - hiện đại (bản đồ độ cao → pháp tuyến → chiếu sáng): `inflated` (chữ
    phồng), `chrome`, `glass` (kính mờ), `holo` (ánh cầu vồng), `gradient`
    (chuyển màu + nhiễu hạt);
  - cổ điển: `metal3d`, `neon`, `candy`, `longshadow`, `outline_pop`.

  Màu: các **màu nổi bật so với tổng thể** của ảnh nền (đèn trên trời đêm,
  bóng bay vàng trên nền hồng); nền không có màu nổi bật thì dùng
  `text_color` / `accent_color` của LLM. `neon`, `glass` không dùng cho nền
  sáng. Kiểu + màu đã chọn ghi vào `final_json.text_effect` và `effect` của
  từng dòng.
- **Ra:** `final_json`, `composite_preview`, `background_treated` (nền đã xoá
  phông — **app tự vẽ chữ thì dùng ảnh này làm nền**), `layout_json`,
  `report` (điểm từng template + mức xoá phông từng dòng).
- Kích thước latent: tính từ tỉ lệ canvas (~1MP, bội 16) như
  `guide.auto_bg_size`; không cần `DesignLayoutEngine` / `DesignZoneCheck`.

### InviteCardPrepare + InviteCardComposite (thiệp mời, chữ trước)

Plan JSON do LLM viết (endpoint Qwen3.5 riêng) → code dựng tờ thiệp ở giữa
có sẵn chữ → Qwen-Image-Edit 2511 inpaint cảnh xung quanh bằng mask mềm →
dán lại nét chữ nguyên pixel. Logic: `designlayout/invite_card.py`; workflow
API mẫu: `examples/invite_card_api.json`.

- **InviteCardPrepare — In:** `plan_json` (câu trả lời của LLM, chấp nhận có
  chữ / code fence bao quanh), `request` (câu yêu cầu gốc: ngày tiếng Việt
  lấy từ đây, kiểm tra câu chép từ ví dụ), `seed`, `width` / `height`
  (mặc định 896×1344), `fonts_dir`, `composition` (`auto` = seed chọn trong
  top 3 LLM xếp hạng). **Ra:** `image` (thiệp + chữ trên nền loang),
  `mask` (0 giữ · 1 vẽ lại → `SetLatentNoiseMask`, model qua
  `DifferentialDiffusion`), `glyph_mask`, `prompt` (cho
  `TextEncodeQwenImageEditPlus`), `report`.
- **InviteCardComposite — In:** `generated` (sau `VAEDecode`), `card_image`,
  `glyph_mask`. **Ra:** ảnh cuối.
- Code tự lo: sửa ngày tiếng Việt (tự tính thứ), bỏ dòng chép ví dụ /
  `<…>`, ngắt dòng tự nhiên, bố cục (classic, modern_line, hero_date,
  save_the_date, editorial) và khung (arch, rounded, ticket) trong top LLM
  xếp hạng, màu giấy giữ dịu / đậm, màu chữ đủ tương phản. Chữ và viền sát
  chữ luôn khoá (mask = 0).

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

`canvas_width/height` quyết định **nhóm tỉ lệ**; cùng với `design_type` nó
chọn ra bộ template phù hợp trong **thư viện ~200 template**
(`designlayout/templates.py`, xem mục *Thư viện template*):

| Nhóm | Tỉ lệ w/h | Ví dụ |
|---|---|---|
| `tall` | < 0.65 | 1080×1920 (story/reels/shorts) |
| `portrait` | 0.65–0.9 | 1080×1350, A4 |
| `square` | 0.9–1.12 | 1080×1080 |
| `landscape` | 1.12–2.1 | 1920×1080, 1280×720 (thumbnail), 1200×628 |
| `wide` | 2.1–4 | 1500×500, 1920×640 (banner) |

Tỉ lệ ngoài 1:4 … 4:1 (vd leaderboard 728×90) **không hỗ trợ**: Layout Engine
báo `ValueError`.

- Template có huy hiệu chỉ được chọn tự động khi có chữ `emphasis` (và được
  cộng trọng số khi có).
- `composition_hint` từ LLM là **nhóm bố cục** (`top, bottom, center,
  top_bottom, column, side, split, badge, card, frame`): chỉ là ưu tiên (x4
  trọng số), engine vẫn chọn template cụ thể theo seed. Tên kiểu cũ
  (`top_headline`, `badge_focus`...) được hiểu là nhóm tương ứng.
- Widget `composition` của node ép đúng một kiểu: tên kiểu cũ hoặc id template
  (vd `wedding.monogram.double_frame`).
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
 "design_type":"poster|banner|thumbnail|invitation|wedding|birthday|social|logo",
 "composition_hint":"auto|top|bottom|center|top_bottom|column|side|split|badge|card|frame",
 "background_prompt":"...","background_color":"#RRGGBB",
 "text_color":"#RRGGBB","accent_color":"#RRGGBB",
 "texts":[{"role":"headline|subheadline|emphasis|body|detail|note","text":"...","priority":1}]}
```

**final_json** giữ tương thích schema cũ (`text, role, font, size_px, cx, cy,
align, color, letter_spacing, shadow, outline_width, outline_color`) + trường
mới tuỳ chọn (`box, lines, line_height, scrim, shapes, canvas, composition,
template, palette, seed`). `palette` = màu LLM gợi ý; `color` của từng dòng là
sắc đó đã chỉnh độ sáng cho đủ tương phản trên ảnh thật. Việc map sang DTO thật của app nằm gọn trong
`designlayout/schema.py::to_app_json` — sửa đúng một chỗ đó khi có định dạng app.

## Thư viện template (~200)

`designlayout/templates.py` khai báo template **dạng dữ liệu**, sinh từ các
dạng gốc × biến thể (căn lề, vị trí, cột, trang trí). Id ổn định, dạng
`<loại>.<dạng>.<biến thể>`:

| Loại | Số template | Ví dụ |
|---|---|---|
| poster | 75 | `poster.stack_top.left.divider`, `poster.bottom_bar.dark.center`, `poster.card.center.light` |
| social | 57 (dùng chung nhiều template poster) | `social.quote.corners`, `social.story_bottom.dark_strip` |
| banner | 40 | `banner.side.left.60.bar`, `banner.badge.middle.star`, `banner.panel_left.accent` |
| thumbnail | 26 | `thumb.big_side.left.outline.star`, `thumb.center_huge.accent_strip` |
| birthday | 22 | `birthday.age_badge.confetti_ribbon.star`, `birthday.card.accent.confetti` |
| wedding | 21 | `wedding.monogram.double_frame`, `wedding.names_top.corners` |
| invitation | 19 | `invite.center.corners`, `invite.card.bottom.dark` |
| logo | 7 | `logo.center.ring`, `logo.stack.ring_divider` |

Mọi template đi qua cùng layout engine nên đều được đảm bảo: không mất/cắt
chữ, không chồng chữ, không tràn khung, đúng thứ bậc cỡ chữ, trang trí không
đè lên chữ (kiểm tra tự động trên 1.350 layout).

**Thêm template:** thêm một lời gọi `_t(...)` trong `templates.py`:

```python
_t("wedding.monogram.my_variant", "frame", ["wedding"], PORTRAITISH,
   primary=(.06, .22, .88, .36), secondary=(.06, .66, .88, .22),  # tỉ lệ vùng an toàn
   decor=D_DOUBLE + D_DIV_DIAMOND,          # khung đôi + đường kẻ hoạ tiết thoi
   panel={"target": "all", "style": "light"})   # tuỳ chọn: thẻ nền sau chữ
```

Tham số khác: `align`, `p_valign`/`s_valign`, `s_align`, `badge=(x,y,w,h)` +
`badge_shape` (`circle|star|diamond|pill`), `ribbon=True` (subheadline trên
ruy băng), `outline` (viền chữ headline kiểu thumbnail, tỉ lệ cỡ chữ),
`type_scale`, `weight` (trọng số chọn tự động). Kiểu cũ trong
`compositions.py` (`BUILDERS`) vẫn giữ để ép bằng tên.

## Shapes trong final_json (app tự vẽ)

Toạ độ theo px canvas, màu hex. Vẽ theo thứ tự trong mảng (tấm nền trước,
chữ vẽ sau cùng). `color_ref` (`text|accent`) chỉ để engine đồng bộ màu, app
dùng `stroke`/`fill` đã tính sẵn.

| type | Trường | Vẽ |
|---|---|---|
| `panel` | `x,y,w,h,fill,opacity,radius` | Hình chữ nhật bo góc, độ trong suốt `opacity` |
| `ribbon` | `x,y,w,h,fill,notch` | Dải ruy băng, 2 đầu khía hình chữ V sâu `notch` |
| `badge` | `shape` = `circle`/`star`/`diamond`: `cx,cy,r,fill`; `pill`: `x,y,w,h,fill` | Huy hiệu đặc; `star` = 14 cánh, bán kính trong 0.82r |
| `border` | `x,y,w,h,stroke,width,style,radius,gap` | `single`; `rounded` (bo `radius`); `double` (khung thứ 2 lùi vào `gap`, nét mảnh bằng nửa) |
| `corners` | `x,y,w,h,len,stroke,width` | 4 góc chữ L, mỗi cạnh dài `len` |
| `divider` | `x0,x1,y,stroke,width,ornament,ornament_size` | Đường ngang; `ornament` `diamond`/`dot`/`dots` ở giữa (cắt đoạn thẳng quanh hoạ tiết) |
| `accent_bar` | `x,y,w,h,fill` | Thanh ngắn bo tròn 2 đầu |
| `dots` | `points:[[x,y,r],...]`, `fills:[hex,...]`, `opacity` | Confetti: hình tròn |
| `ring` | `cx,cy,r,stroke,width` | Vòng tròn viền |
| `line` | `x0,y0,x1,y1,stroke,width` | Đường thẳng (kiểu cũ) |

Chữ có `on_shape: true` nằm trên huy hiệu/ruy băng/tấm nền màu nhấn (màu đã
khớp với shape); `on_panel` = nằm trên tấm nền sáng/tối.

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
