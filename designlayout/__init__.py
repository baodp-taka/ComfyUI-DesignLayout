"""ComfyUI-DesignLayout core library (pure Python, no ComfyUI import).

Split into small modules so the whole layout pipeline is testable offline
without a GPU or ComfyUI running:

    parsing        robust LLM-JSON parsing + background-prompt cleaning
    fonts          font registry, glyph-coverage check, language/style pick
    measure        real-font text measuring, Vietnamese-aware wrapping, fit
    compositions   named layout templates (top_headline, bottom_band, ...)
    layout         the layout engine: place text blocks on a 12-col grid
    guide          guide image + text mask + augmented background prompt
    zone_check     busyness / palette / WCAG contrast post-check on the render
    render_preview debug + composite previews (real text drawn with real fonts)
    schema         shared constants, defaults, final_json emitter
    tensors        PIL <-> torch [B,H,W,C] / MASK [B,H,W] helpers
"""

__version__ = "0.1.0"
