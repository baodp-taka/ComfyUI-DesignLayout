# -*- coding: utf-8 -*-
"""SaveTextFile + per-node loading (offline: a fake folder_paths, no ComfyUI)."""
import importlib.util
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_folder_paths(tmp):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp)

    def get_save_image_path(prefix, out_dir):
        return str(out_dir), prefix, 1, "", prefix
    fp.get_save_image_path = get_save_image_path
    return fp


def test_save_text_file_lists_the_file_as_an_output(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", _fake_folder_paths(tmp_path))
    spec = importlib.util.spec_from_file_location("text_node", os.path.join(ROOT, "nodes", "text_node.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = {"names": "Hoàng Nam và Bảo Anh"}
    out = mod.SaveTextFile().save(json.dumps(plan, ensure_ascii=False), "plan", "json")
    entry = out["ui"]["images"][0]
    assert entry == {"filename": "plan_00001_.json", "subfolder": "", "type": "output"}
    with open(tmp_path / entry["filename"], encoding="utf-8") as f:
        assert json.load(f) == plan
    assert out["result"] == ()


def test_one_node_failing_does_not_hide_the_others():
    spec = importlib.util.spec_from_file_location(
        "designlayout_pkg", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["designlayout_pkg"] = pkg
    try:
        spec.loader.exec_module(pkg)
    finally:
        sys.modules.pop("designlayout_pkg", None)
    # offline there is no torch: the image nodes may fail, SaveTextFile must not
    assert "SaveTextFile" in pkg.NODE_CLASS_MAPPINGS
