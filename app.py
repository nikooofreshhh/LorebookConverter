import json
from typing import Any, Dict, List, Set, Tuple
from copy import deepcopy
import re
from html import escape

import streamlit as st
import colorsys

# Simple converter mapping (compatible with previous map_entry)
def map_entry(source_entry: Dict[str, Any], uid: int, order_value: int) -> Dict[str, Any]:
    keys = [k.get("keyText", "") for k in source_entry.get("keys", [])]
    description = source_entry.get("description", "") or ""
    name = source_entry.get("name", "")

    mapped = {
        "uid": uid,
        "key": keys,
        "keysecondary": [],
        "comment": name,
        "content": "\n" + description + "\n",
        "constant": False,
        "selective": True,
        "order": order_value,
        "position": 0,
        "disable": False,
        "displayIndex": uid,
        "addMemo": True,
        "group": "",
        "groupOverride": False,
        "groupWeight": 100,
        "sticky": 0,
        "cooldown": 0,
        "delay": 0,
        "probability": 100,
        "depth": 4,
        "useProbability": True,
        "role": None,
        "vectorized": False,
        "excludeRecursion": False,
        "preventRecursion": False,
        "delayUntilRecursion": False,
        "scanDepth": None,
        "caseSensitive": None,
        "matchWholeWords": None,
        "useGroupScoring": None,
        "automationId": "",
        "selectiveLogic": 0,
        "ignoreBudget": False,
        "matchPersonaDescription": False,
        "matchCharacterDescription": False,
        "matchCharacterPersonality": False,
        "matchCharacterDepthPrompt": False,
        "matchScenario": False,
        "matchCreatorNotes": False,
        "outletName": "",
        "triggers": [],
        "characterFilter": {"isExclude": False, "names": [], "tags": []},
    }

    return mapped


# Helper: full-word, case-insensitive matches
def find_key_positions(text: str, key: str) -> List[Tuple[int, int]]:
    if not key:
        return []
    try:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", flags=re.IGNORECASE)
    except re.error:
        matches = []
        idx = text.lower().find(key.lower())
        while idx != -1:
            matches.append((idx, idx + len(key)))
            idx = text.lower().find(key.lower(), idx + 1)
        return matches
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def build_trigger_edges(entries: List[Dict[str, Any]]) -> List[Tuple[int, int, List[str]]]:
    # returns list of (from_idx, to_idx, [matching keys])
    edges: List[Tuple[int, int, List[str]]] = []
    n = len(entries)
    key_map: List[List[str]] = []
    for e in entries:
        key_map.append([k.get("keyText", "") for k in e.get("keys", [])])

    for i, e in enumerate(entries):
        desc = e.get("description", "") or ""
        for j in range(n):
            if i == j:
                continue
            matches = []
            for key in key_map[j]:
                if find_key_positions(desc, key):
                    matches.append(key)
            if matches:
                edges.append((i, j, matches))
    return edges


def highlight_text(text: str, matches: List[Tuple[int, int]]) -> str:
    if not matches:
        return escape(text)
    matches_sorted = sorted(matches)
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = matches_sorted[0]
    for s, e in matches_sorted[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    parts: List[str] = []
    last = 0
    for s, e in merged:
        parts.append(escape(text[last:s]))
        parts.append(f"<mark>{escape(text[s:e])}</mark>")
        last = e
    parts.append(escape(text[last:]))
    return "".join(parts)


def _assign_colors(n: int) -> List[str]:
    # return n visually distinct hex colors
    colors: List[str] = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.95)
        colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
    return colors


def highlight_with_colors(text: str, colored_matches: List[Tuple[int, int, str]]) -> str:
    # colored_matches: list of (start, end, hexcolor). Non-overlapping output; overlapping matches prefer earlier ones.
    if not colored_matches:
        return escape(text)
    # sort by start, longer first
    colored_matches_sorted = sorted(colored_matches, key=lambda x: (x[0], -(x[1] - x[0])))
    parts: List[str] = []
    last = 0
    last_end = 0
    for s, e, color in colored_matches_sorted:
        if s < last_end:
            continue
        parts.append(escape(text[last:s]))
        parts.append(f"<span style=\"background-color:{color};padding:2px;border-radius:3px\">{escape(text[s:e])}</span>")
        last = e
        last_end = e
    parts.append(escape(text[last:]))
    return "".join(parts)


def reset_state_for_new_import() -> None:
    """Clear session state keys that should not persist across imports."""
    to_clear_prefixes = [
        "entries_original",
        "entries_imported",
        "entries_work",
        "snipped_keys",
        "order",
        "initial_order",
        "initial_order_imported",
        "use_session_entries",
        "inspect_selected",
        "editing_idx",
        "edit_keys_csv_",
        "edit_text_",
        "edge_key_",
        "reorder_",
        "expand_targets",
        "reorder_show_bulk",
        "pasted_to_import",
        "pasted_text_area",
        "file_uploader",
        "undo_stack"
    ]
    keys = list(st.session_state.keys())
    for k in keys:
        for p in to_clear_prefixes:
            if k == p or k.startswith(p):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
                break


def reset_all_changes(entries_len: int) -> None:
    """Reset snips, order, and edits back to the original imported entries."""
    st.session_state.undo_stack = []
    st.session_state.snipped_keys = {}
    imported = st.session_state.get("entries_imported")
    if imported is not None:
        st.session_state.entries_original = deepcopy(imported)
        st.session_state.entries_work = deepcopy(imported)
        entries_len = len(imported)
    st.session_state.initial_order = _reconcile_order(
        st.session_state.get("initial_order_imported", list(range(entries_len))),
        entries_len,
    )
    st.session_state.order = st.session_state.initial_order
    st.session_state.use_session_entries = True
    if "editing_idx" in st.session_state:
        del st.session_state["editing_idx"]
    for k in list(st.session_state.keys()):
        if k.startswith("edit_"):
            try:
                del st.session_state[k]
            except Exception:
                pass


def push_undo() -> None:
    """Snapshot current mutable state onto the undo stack."""
    snapshot = {
        "entries_work": deepcopy(st.session_state.get("entries_work", [])),
        "entries_original": deepcopy(st.session_state.get("entries_original", [])),
        "snipped_keys": deepcopy(st.session_state.get("snipped_keys", {})),
        "order": list(st.session_state.get("order", [])),
        "initial_order": list(st.session_state.get("initial_order", [])),
    }
    stack = st.session_state.get("undo_stack", [])
    stack.append(snapshot)
    st.session_state.undo_stack = stack[-20:]  # cap at 20


def detect_format(raw_entries: Any) -> str:
    """Auto-detect format: 'DreamJourney' or 'SillyTavern'."""
    if isinstance(raw_entries, dict):
        # SillyTavern uses dict keyed by string indices
        sample = next(iter(raw_entries.values())) if raw_entries else {}
        if "comment" in sample or "content" in sample or "key" in sample:
            return "SillyTavern"
    elif isinstance(raw_entries, list) and raw_entries:
        sample = raw_entries[0]
        if "name" in sample or "description" in sample or "keys" in sample:
            return "DreamJourney"
    return None  # unknown format


def normalize_entry(entry: Dict[str, Any], detected_format: str) -> Dict[str, Any]:
    """Map entry from SillyTavern or DreamJourney to universal internal format."""
    if detected_format == "SillyTavern":
        name = entry.get("comment") or entry.get("name", "")
        desc = entry.get("content", "") or ""
        # strip surrounding newlines if present
        if isinstance(desc, str) and desc.startswith("\n") and desc.endswith("\n"):
            desc = desc[1:-1]
        keys = entry.get("key", []) or []
        key_objs = [{"keyText": k} for k in keys] if isinstance(keys, list) else []
        entry_type = "other"  # SillyTavern doesn't have type, so default to "other"
    else:
        # DreamJourney format
        name = entry.get("name", "")
        desc = entry.get("description", "") or ""
        key_objs = entry.get("keys", []) or []
        entry_type = entry.get("type", "other")  # Preserve type from DreamJourney
    
    return {
        "name": name,
        "description": desc,
        "keys": key_objs,
        "type": entry_type
    }


def normalize_entries(raw_entries: Any, detected_format: str) -> List[Dict[str, Any]]:
    """Convert raw entries (list or dict) to list of normalized internal format."""
    # Convert dict to list if needed
    if isinstance(raw_entries, dict):
        entries_list = list(raw_entries.values())
    else:
        entries_list = raw_entries if isinstance(raw_entries, list) else []
    
    # Normalize each entry
    return [normalize_entry(e, detected_format) for e in entries_list]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reconcile_order(order_list: List[int], entries_len: int) -> List[int]:
    """Ensure order list contains each index 0..entries_len-1 exactly once, preserving existing order."""
    seen: Set[int] = set()
    cleaned: List[int] = []
    for idx in order_list:
        if isinstance(idx, int) and 0 <= idx < entries_len and idx not in seen:
            cleaned.append(idx)
            seen.add(idx)
    for idx in range(entries_len):
        if idx not in seen:
            cleaned.append(idx)
    return cleaned


def build_initial_order(raw_entries: Any, detected_format: str) -> List[int]:
    """Build initial order mapping (output position -> source index) for import."""
    if isinstance(raw_entries, dict):
        entries_list = list(raw_entries.values())
    else:
        entries_list = raw_entries if isinstance(raw_entries, list) else []

    if detected_format != "SillyTavern":
        return list(range(len(entries_list)))

    keyed_indices: List[Tuple[int, Tuple[int, int]]] = []
    for idx, entry in enumerate(entries_list):
        order_value = _safe_int(entry.get("order"), idx)
        keyed_indices.append((idx, (order_value, idx)))

    keyed_indices.sort(key=lambda item: item[1])
    return [idx for idx, _ in keyed_indices]


def export_to_format(entries_work: List[Dict[str, Any]], snipped_keys: Dict[int, Set[str]], order: List[int], export_format: str, lorebook_name: str) -> Dict[str, Any]:
    """Export entries in the specified format (SillyTavern or DreamJourney)."""
    exported_entries: Dict[str, Any] = {}
    
    for out_uid, src_idx in enumerate(order):
        src = entries_work[src_idx]
        order_value = out_uid + 1
        mapped = map_entry(src, out_uid, order_value)
        
        # filter snipped keys
        if src_idx in snipped_keys:
            snipped = snipped_keys[src_idx]
            mapped_keys = [k for k in mapped.get("key", []) if k not in snipped]
            mapped["key"] = mapped_keys
        
        exported_entries[str(out_uid)] = mapped
    
    if export_format == "DreamJourney":
        # Convert to DreamJourney format (list of entries with name/description/keys/type)
        entries_list = []
        for idx in order:
            src = entries_work[idx]
            keys_list = src.get("keys", []) or []  # Already in {"keyText": "..."} format
            # Filter out snipped keys
            if idx in snipped_keys:
                snipped_set = snipped_keys[idx]
                keys_list = [k for k in keys_list if k.get('keyText', '') not in snipped_set]
            
            entry_type = src.get("type", "other")
            entries_list.append({
                "name": src.get("name", ""),
                "description": src.get("description", ""),
                "type": entry_type,
                "keys": keys_list
            })
        return {
            "name": lorebook_name,
            "isPublic": False,
            "thumbnail": None,
            "entries": entries_list
        }
    else:
        # SillyTavern format (dict with full metadata)
        return {"entries": exported_entries}
    
def clear_text_box(key: str) -> None:
    """Utility to clear a text area or input by key."""
    if key in st.session_state:
        try:
            st.session_state[key] = ""
            st.session_state.pop('pasted_to_import', None)
        except Exception:
            pass


def _parse_keys_csv(keys_text: str) -> List[Dict[str, str]]:
    keys = [k.strip() for k in re.split(r"[,\n]", keys_text) if k.strip()]
    return [{"keyText": k} for k in keys]


def main():
    st.set_page_config(page_title="Niko's Lorebook Tool", layout="wide")
    st.title("Lorebook Tool")
    st.markdown("Upload a SillyTavern or DreamJourney lorebook. Convert formats, reorder entries, and clean up cascades. By NikoFresh <3")

    uploaded = st.file_uploader("Upload source JSON", type=["json"],key="file_uploader")
    pasted = st.text_area("Or paste source JSON", value="", height=120,key="pasted_text_area")

    # Only import pasted text when the user clicks the button
    btn_col_1, btn_col_2, btn_col_3, btn_col_4 = st.columns([1, 1, 2, 2])
    with btn_col_1:
        if st.button("Import pasted JSON", key="import_pasted_btn"):
            st.session_state['pasted_to_import'] = pasted
            st.rerun()
    with btn_col_2:
        if st.button("Start from blank", key="start_from_blank_btn"):
            reset_state_for_new_import()
            st.session_state.use_session_entries = True
            st.session_state.entries_original = []
            st.session_state.entries_imported = []
            st.session_state.entries_work = []
            st.session_state.snipped_keys = {}
            st.session_state.initial_order = []
            st.session_state.initial_order_imported = []
            st.session_state.order = []
            st.rerun()
    with btn_col_3:
        st.button("Clear pasted JSON", key="clear_pasted_btn", on_click=clear_text_box, args=("pasted_text_area",))
    with btn_col_4:
        if st.button("New Import (clear state after prior import)", key="new_import_btn"):
            reset_state_for_new_import()
            st.rerun()

    # Determine which source to parse: uploaded file or previously-stashed pasted text
    data = None
    use_session_entries = st.session_state.get("use_session_entries", False)
    if uploaded is not None:
        try:
            data = json.load(uploaded)
        except Exception as e:
            st.error(f"Could not parse uploaded JSON: {e}")
            return
    elif st.session_state.get('pasted_to_import') is not None:
        try:
            data = json.loads(st.session_state.get('pasted_to_import'))
        except Exception as e:
            st.error(f"Could not parse pasted JSON: {e}")
            return
    elif use_session_entries:
        data = {"entries": []}
    else:
        st.info("Upload a file or paste JSON and click 'Import pasted JSON' to begin.")
        return

    raw_entries = data.get("entries", [])

    # Auto-detect format
    detected_format = detect_format(raw_entries)

    # If auto-detection fails, ask user; otherwise use detected format
    if detected_format:
        st.sidebar.markdown(f"**Format auto-detected: {detected_format}**")
        # Still allow override if needed
        if st.sidebar.checkbox("Override format detection?", value=False):
            import_format = st.sidebar.selectbox("Choose import format", ["DreamJourney", "SillyTavern"], index=0)
        else:
            import_format = detected_format
    else:
        st.sidebar.markdown("**Format**")
        import_format = st.sidebar.selectbox("Choose import format", ["DreamJourney", "SillyTavern"], index=0)

    export_format = st.sidebar.selectbox(
        "Export file format",
        ["SillyTavern", "DreamJourney"],
        index=0,
        key="export_format",
    )

    # Normalize entries to universal internal format
    if use_session_entries and "entries_work" in st.session_state:
        entries = st.session_state["entries_work"]
        if "initial_order" not in st.session_state or len(st.session_state.get("initial_order", [])) != len(entries):
            base_initial = st.session_state.get("initial_order", st.session_state.get("order", list(range(len(entries)))))
            st.session_state.initial_order = _reconcile_order(base_initial, len(entries))
    else:
        entries = normalize_entries(raw_entries, import_format)
        initial_order = build_initial_order(raw_entries, import_format)

        # keep a mutable copy of entries in session state so edits persist across reruns
        if "entries_original" not in st.session_state or len(st.session_state.get("entries_original", [])) != len(entries):
            st.session_state["entries_original"] = deepcopy(entries)
        if "entries_imported" not in st.session_state or len(st.session_state.get("entries_imported", [])) != len(entries):
            st.session_state["entries_imported"] = deepcopy(entries)
        if "entries_work" not in st.session_state or len(st.session_state.get("entries_work", [])) != len(entries):
            st.session_state["entries_work"] = deepcopy(entries)
        entries = st.session_state["entries_work"]

        if "initial_order" not in st.session_state or len(st.session_state.get("initial_order", [])) != len(entries):
            st.session_state.initial_order = initial_order
        if "initial_order_imported" not in st.session_state or len(st.session_state.get("initial_order_imported", [])) != len(entries):
            st.session_state.initial_order_imported = initial_order

    # session state
    if "snipped_keys" not in st.session_state:
        st.session_state.snipped_keys = {}  # {entry_idx: set(keys)}
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []

    # Reorder UI: maintain an `order` mapping (output position -> source index)
    if "initial_order" not in st.session_state or len(st.session_state.get("initial_order", [])) != len(entries):
        st.session_state.initial_order = _reconcile_order(st.session_state.get("initial_order", list(range(len(entries)))), len(entries))
    if "order" not in st.session_state or len(st.session_state.get("order", [])) != len(entries):
        st.session_state.order = _reconcile_order(st.session_state.get("order", st.session_state.initial_order), len(entries))

    display_order = st.session_state.get("order", list(range(len(entries))))

    st.header("Add/Delete Entry")
    with st.expander("Add a new entry", expanded=False):
        with st.form("add_entry_form", clear_on_submit=True):
            new_name = st.text_input("Name", value="")
            new_desc = st.text_area("Description", value="", height=140)
            new_keys = st.text_area("Keys (comma or newline separated)", value="", height=80)
            new_type = st.selectbox("Type (for DreamJourney)", ["character", "object", "plot", "other"], index=3)
            position_value = st.number_input(
                "Position (1 = top)",
                min_value=1,
                max_value=len(entries) + 1,
                value=len(entries) + 1,
                step=1,
            )
            submitted = st.form_submit_button("Add entry")

        if submitted:
            if not new_name and not new_desc and not new_keys:
                st.warning("Please provide at least a name, description, or key before adding.")
            else:
                push_undo()
                current_order = st.session_state.get("order", list(range(len(entries))))
                current_initial = st.session_state.get("initial_order", list(range(len(entries))))
                entry_obj = {
                    "name": new_name.strip(),
                    "description": new_desc.strip(),
                    "keys": _parse_keys_csv(new_keys),
                    "type": new_type,
                }
                st.session_state.entries_original.append(deepcopy(entry_obj))
                st.session_state.entries_work.append(entry_obj)
                st.session_state.use_session_entries = True

                new_idx = len(st.session_state.entries_work) - 1
                insert_at = max(0, min(len(current_order), int(position_value) - 1))
                current_order.insert(insert_at, new_idx)
                current_initial.insert(insert_at, new_idx)

                st.session_state.initial_order = current_initial
                st.session_state.order = current_order
                st.rerun()

    with st.expander("Delete an entry", expanded=False):
        if not entries:
            st.info("No entries to delete.")
        else:
            delete_options = [f"{pos + 1}. {entries[idx].get('name','(no name)')}" for pos, idx in enumerate(display_order)]
            delete_choice = st.selectbox("Select entry to delete", delete_options, index=0, key="delete_entry_select")
            delete_pos = delete_options.index(delete_choice)
            delete_idx = display_order[delete_pos]
            confirm_delete = st.checkbox("Confirm delete", value=False, key="delete_entry_confirm")
            if st.button("Delete entry", key="delete_entry_btn"):
                if not confirm_delete:
                    st.warning("Please confirm delete before proceeding.")
                else:
                    push_undo()
                    st.session_state.entries_work.pop(delete_idx)
                    if "entries_original" in st.session_state and len(st.session_state.entries_original) > delete_idx:
                        st.session_state.entries_original.pop(delete_idx)

                    new_snipped = {}
                    for k, v in st.session_state.get("snipped_keys", {}).items():
                        if k == delete_idx:
                            continue
                        new_snipped[k - 1 if k > delete_idx else k] = v
                    st.session_state.snipped_keys = new_snipped

                    def _remap_order(order_list: List[int]) -> List[int]:
                        remapped = []
                        for idx in order_list:
                            if idx == delete_idx:
                                continue
                            remapped.append(idx - 1 if idx > delete_idx else idx)
                        return remapped

                    st.session_state.order = _remap_order(st.session_state.get("order", []))
                    st.session_state.initial_order = _remap_order(st.session_state.get("initial_order", []))
                    st.session_state.use_session_entries = True
                    st.rerun()

    st.header("Reorder Entries")
    with st.expander("Reorder entries (optional, changes order of entry placement in output)", expanded=False):
        if not entries:
            st.info("No entries yet. Add an entry to enable reordering.")
        else:
            st.caption("Select an entry, preview it, and use the controls to move it. For bulk edits, paste a new numerical order below.")

            # Display select + preview + controls
            options_display = [f"{pos+1}. {entries[idx].get('name','(no name)')}" for pos, idx in enumerate(st.session_state.order)]
            selected = st.selectbox("Select entry to edit order", options_display, index=0, key="reorder_select_entry")
            sel_pos = options_display.index(selected)
            sel_idx = st.session_state.order[sel_pos]

            # Preview description
            with st.expander("Preview entry description", expanded=False):
                desc = entries[sel_idx].get("description", "") or ""
                st.write(desc)

            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button("Move Up", key="reorder_move_up"):
                    if sel_pos > 0:
                        push_undo()
                        o = st.session_state.order
                        o[sel_pos - 1], o[sel_pos] = o[sel_pos], o[sel_pos - 1]
                        st.session_state.order = o
                        st.rerun()
            with cols[1]:
                if st.button("Move Down", key="reorder_move_down"):
                    if sel_pos < len(st.session_state.order) - 1:
                        push_undo()
                        o = st.session_state.order
                        o[sel_pos + 1], o[sel_pos] = o[sel_pos], o[sel_pos + 1]
                        st.session_state.order = o
                        st.rerun()
            with cols[2]:
                new_pos = st.number_input("Move to position (1 = top)", min_value=1, max_value=len(entries), value=sel_pos + 1, step=1, key="reorder_move_to_pos")
                if st.button("Apply Move", key="reorder_apply_move"):
                    push_undo()
                    cur = [x for x in st.session_state.order if x != sel_idx]
                    insert_at = max(0, min(len(cur), new_pos - 1))
                    cur.insert(insert_at, sel_idx)
                    st.session_state.order = cur
                    st.rerun()

            st.markdown("---")

            # Reset and Reverse buttons at top
            cols_top = st.columns([1, 1])
            with cols_top[0]:
                if st.button("Reset order", key="reorder_reset_order"):
                    push_undo()
                    st.session_state.order = st.session_state.get("initial_order", list(range(len(entries))))
                    st.rerun()
            with cols_top[1]:
                if st.button("Reverse order", key="reorder_reverse_order"):
                    push_undo()
                    st.session_state.order = list(reversed(st.session_state.get("order", list(range(len(entries))))))
                    st.rerun()

            st.markdown("---")

            # Quick reorder list view (expanded by default)
            with st.expander("Quick reorder list", expanded=True):
                st.caption("Click up/down buttons to reorder entries quickly.")
                for pos, idx in enumerate(st.session_state.order):
                    entry_name = entries[idx].get('name', '(no name)')
                    cols_list = st.columns([0.5, 1, 3, 0.5, 0.5])
                    with cols_list[0]:
                        st.markdown(f"**{pos+1}**")
                    with cols_list[1]:
                        st.markdown(f"`{idx}`")
                    with cols_list[2]:
                        st.markdown(entry_name)
                    with cols_list[3]:
                        if st.button("↑", key=f"quick_move_up_{pos}", help="Move up"):
                            if pos > 0:
                                push_undo()
                                o = st.session_state.order
                                o[pos - 1], o[pos] = o[pos], o[pos - 1]
                                st.session_state.order = o
                                st.rerun()
                    with cols_list[4]:
                        if st.button("↓", key=f"quick_move_down_{pos}", help="Move down"):
                            if pos < len(st.session_state.order) - 1:
                                push_undo()
                                o = st.session_state.order
                                o[pos + 1], o[pos] = o[pos], o[pos + 1]
                                st.session_state.order = o
                                st.rerun()

            # Bulk reorder (collapsed by default)
            with st.expander("Bulk reorder", expanded=False):
                st.caption("Paste one line per entry in the desired order. Lines can be the displayed 'N. name' strings, or the original source index numbers (0-based).")
                bulk = st.text_area("Paste new order here", value="", height=120, placeholder="e.g.\n0\n2\n1\n... OR copy the displayed lines exactly")
                if st.button("Apply bulk order", key="reorder_apply_bulk"):
                    lines = [l.strip() for l in bulk.splitlines() if l.strip()]
                    if len(lines) != len(entries):
                        st.error(f"Bulk list length {len(lines)} doesn't match number of entries {len(entries)}")
                    else:
                        parsed = []
                        ok = True
                        for ln in lines:
                            if ln.isdigit():
                                parsed.append(int(ln))
                            else:
                                try:
                                    idx = options_display.index(ln)
                                    parsed.append(st.session_state.order[idx])
                                except ValueError:
                                    st.error(f"Could not parse line: '{ln}'")
                                    ok = False
                                    break
                        if ok:
                            push_undo()
                            st.session_state.order = parsed
                            st.rerun()

                if "reorder_show_bulk" not in st.session_state:
                    st.session_state.reorder_show_bulk = False
                btn_label = "Hide" if st.session_state.reorder_show_bulk else "Show current order in bulk format"
                if st.button(btn_label, key="reorder_toggle_bulk"):
                    st.session_state.reorder_show_bulk = not st.session_state.reorder_show_bulk
                    st.rerun()
                if st.session_state.get("reorder_show_bulk", False):
                    st.code("\n".join(str(i) for i in st.session_state.order))

    
    # Move sidebar controls down - put export options first, then actions at the bottom
    st.sidebar.markdown("## Export Settings")
    lorebook_name = st.sidebar.text_input("Lorebook name", value="MyLorebook", key="lorebook_name_input")
    
    st.sidebar.markdown("---")
    # Build output with snips applied for immediate download
    export_data = export_to_format(
        entries,
        st.session_state.get("snipped_keys", {}),
        st.session_state.get("order", list(range(len(entries)))),
        st.session_state.get("export_format", "SillyTavern"),
        lorebook_name
    )
    json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode('utf-8')
    st.sidebar.download_button(
        "Convert & Download",
        data=json_bytes,
        file_name=f"{lorebook_name}.json",
        mime="application/json"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Reset All Changes (Can't Be Undone!!)")
    confirm_reset = st.sidebar.checkbox("Confirm reset", value=False, key="reset_all_confirm")
    if st.sidebar.button("Reset all", key="sidebar_reset_all"):
        if not confirm_reset:
            st.sidebar.warning("Please confirm reset before proceeding.")
        else:
            reset_all_changes(len(entries))
            st.rerun()

    st.sidebar.markdown("---")
    if st.session_state.get("undo_stack"):
        if st.sidebar.button(f"Undo last action ({len(st.session_state.undo_stack)} in history)", key="sidebar_undo"):
            snapshot = st.session_state.undo_stack.pop()
            st.session_state.entries_work = snapshot["entries_work"]
            st.session_state.entries_original = snapshot["entries_original"]
            st.session_state.snipped_keys = snapshot["snipped_keys"]
            st.session_state.order = snapshot["order"]
            st.session_state.initial_order = snapshot["initial_order"]
            st.rerun()
    else:
        st.sidebar.button("Undo last action (nothing to undo)", key="sidebar_undo_disabled", disabled=True)

    st.header("Cascade Cleanup")
    if not entries:
        st.info("No entries to analyze for cascade cleanup.")
        return
    edges = build_trigger_edges(entries)

    # Build display list excluding edges with all keys already snipped
    display_edges = []
    for idx, (u, v, keys) in enumerate(edges):
        already = st.session_state.get("snipped_keys", {}).get(v, set())
        remaining = [k for k in keys if k not in already]
        if remaining:
            display_edges.append((idx, u, v, remaining))

    total_edges = sum(len(keys) if isinstance(keys, list) else 0 for (_, _, _, keys) in display_edges)

    # group by parent (u) for the snip UI
    parent_map: Dict[int, List[Tuple[int, int, List[str]]]] = {}
    for (_, u, v, keys) in display_edges:
        parent_map.setdefault(u, []).append((u, v, keys))

    # Wrap quick inspect and snip UI under one collapsed section
    with st.expander("Cascade Cleanup (optional, assists in identifying and changing entries that cause cascades)", expanded=False):
        # Reset snips button specific to this section
        col_reset_snips, col_reset_edits = st.columns([1, 1])
        if col_reset_snips.button("Reset snips", key="cascade_reset_snips"):
            push_undo()
            st.session_state.snipped_keys = {}
            st.rerun()
        if col_reset_edits.button("Reset edits", key="cascade_reset_edits"):
            push_undo()
            # Restore edited entries to original values and clear edit UI state
            if "entries_original" in st.session_state:
                st.session_state.entries_work = deepcopy(st.session_state.entries_original)
            if "editing_idx" in st.session_state:
                del st.session_state["editing_idx"]
            for k in list(st.session_state.keys()):
                if k.startswith("edit_"):
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
            st.rerun()
        st.subheader("Inspect Entry - Cascades Highlighted")
        st.markdown("Select an entry. Words in the body of the entry that appear in the key list (or body) of other entries will be automatically highlighted and color coded.\n Entries can be edited, or connections can be snipped below.")
        trace_options = [f"{pos + 1}. {entries[idx].get('name','(no name)')}" for pos, idx in enumerate(display_order)]
        if "inspect_selected" not in st.session_state:
            st.session_state["inspect_selected"] = trace_options[0]
        # Ensure the selected value is still valid
        if st.session_state["inspect_selected"] not in trace_options:
            st.session_state["inspect_selected"] = trace_options[0]
        trace_selected = st.selectbox("Select entry to inspect", trace_options, key="inspect_selected")
        trace_pos = trace_options.index(trace_selected)
        trace_idx = display_order[trace_pos]
        traced_desc = entries[trace_idx].get('description','') or ''

        # find outgoing first-level targets from this entry
        children = []
        for (u, v, keys) in edges:
            if u != trace_idx:
                continue
            already = st.session_state.get('snipped_keys', {}).get(v, set())
            remaining = [k for k in keys if k not in already]
            if remaining:
                children.append((u, v, remaining))

        if not children:
            # fallback: highlight occurrences of any other entries' keys in this entry
            matches: List[Tuple[int, int]] = []
            for j in range(len(entries)):
                if j == trace_idx:
                    continue
                for key in [k.get('keyText','') for k in entries[j].get('keys', [])]:
                    # skip if snipped for traced entry
                    if key in st.session_state.get('snipped_keys', {}).get(trace_idx, set()):
                        continue
                    for (s, t) in find_key_positions(traced_desc, key):
                        matches.append((s, t))
            if matches:
                st.markdown(highlight_text(traced_desc, matches), unsafe_allow_html=True)
            else:
                st.write(traced_desc)

        # Edit entry UI: include comma-separated keys input
        edit_key = f"editing_idx"
        edit_keys_csv = f"edit_keys_csv_{trace_idx}"
        if st.button("Edit entry", key=f"edit_btn_{trace_idx}"):
            st.session_state[edit_key] = trace_idx
            # initialize keys excluding any snipped keys for this entry
            current_keys = [k.get('keyText','') for k in entries[trace_idx].get('keys', [])]
            snipped = st.session_state.get('snipped_keys', {}).get(trace_idx, set())
            filtered = [k for k in current_keys if k not in snipped]
            st.session_state[edit_keys_csv] = ", ".join(filtered)
            st.rerun()

        if st.session_state.get(edit_key) == trace_idx:
            # show editable text area with current description
            edit_area_key = f"edit_text_{trace_idx}"
            new_text = st.text_area("Edit description (changes saved when you click Save)", value=entries[trace_idx].get("description", ""), key=edit_area_key, height=240)

            st.markdown("**Edit keys (comma-separated). Keep commas between each key.**")
            # apply any pending update trigger before widget creation
            trigger_key = f"edit_update_trigger_{trace_idx}"
            if trigger_key in st.session_state:
                st.session_state[edit_keys_csv] = st.session_state.pop(trigger_key)
            # Avoid widget key warning: don't pass value param to text_input if key already exists in session_state
            new_keys_csv = st.text_input("Keys (comma-separated)", key=edit_keys_csv)

            # show highlighted preview of remaining triggers in the edited text
            preview_matches: List[Tuple[int, int]] = []
            for j in range(len(entries)):
                if j == trace_idx:
                    continue
                for key in [k.get('keyText','') for k in entries[j].get('keys', [])]:
                    if key in st.session_state.get('snipped_keys', {}).get(trace_idx, set()):
                        continue
                    for (s, t) in find_key_positions(new_text, key):
                        preview_matches.append((s, t))

            st.markdown("**Preview (highlights remaining trigger keys in your edited text):**")
            if preview_matches:
                st.markdown(highlight_text(new_text, preview_matches), unsafe_allow_html=True)
            else:
                st.write(new_text)

            col_save, col_update, col_cancel = st.columns([1, 1, 1])
            if col_save.button("Save", key=f"save_edit_{trace_idx}"):
                push_undo()
                # persist description and keys
                st.session_state['entries_work'][trace_idx]['description'] = new_text
                # parse CSV into list of keys
                parsed_keys = [k.strip() for k in new_keys_csv.split(',') if k and k.strip()]
                st.session_state['entries_work'][trace_idx]['keys'] = [{'keyText': k} for k in parsed_keys]
                st.success("Saved changes to entry description and keys.")
                # cleanup
                if edit_key in st.session_state:
                    del st.session_state[edit_key]
                if edit_keys_csv in st.session_state:
                    del st.session_state[edit_keys_csv]
                st.rerun()
            if col_update.button("Update Preview", key=f"update_preview_{trace_idx}"):
                # Prepare updated CSV in a trigger value and rerun so we can set the widget-backed key BEFORE the widget is created
                edit_keys_csv_local = f"edit_keys_csv_{trace_idx}"
                current_keys = [k.get('keyText','') for k in entries[trace_idx].get('keys', [])]
                snipped_local = st.session_state.get('snipped_keys', {}).get(trace_idx, set())
                filtered_local = [k for k in current_keys if k not in snipped_local]
                trig_key = f"edit_update_trigger_{trace_idx}"
                st.session_state[trig_key] = ", ".join(filtered_local)
                st.rerun()
            if col_cancel.button("Cancel", key=f"cancel_edit_{trace_idx}"):
                if edit_key in st.session_state:
                    del st.session_state[edit_key]
                if edit_keys_csv in st.session_state:
                    del st.session_state[edit_keys_csv]
                st.rerun()
        else:
            # assign colors for children
            colors = _assign_colors(len(children))

            # build colored matches for the parent (traced) description
            parent_colored: List[Tuple[int, int, str]] = []
            for idx, (u, v, keys) in enumerate(children):
                color = colors[idx]
                for key in keys:
                    for (s, t) in find_key_positions(traced_desc, key):
                        parent_colored.append((s, t, color))

            st.markdown(highlight_with_colors(traced_desc, parent_colored), unsafe_allow_html=True)

            # controls to expand/collapse child details
            if 'expand_targets' not in st.session_state:
                st.session_state.expand_targets = True
            c1, c2 = st.columns([1, 1])
            if c1.button('Collapse all targets'):
                st.session_state.expand_targets = False
                st.rerun()
            if c2.button('Expand all targets'):
                st.session_state.expand_targets = True
                st.rerun()

            # show each child with matching keys highlighted in same color
            for idx, (u, v, keys) in enumerate(children):
                color = colors[idx]
                child_label = f"{v}. {entries[v].get('name','(no name)')} — keys: {', '.join(keys)}"
                with st.expander(child_label, expanded=st.session_state.expand_targets):
                    # show keys with color highlights
                    key_html_parts: List[str] = []
                    for key in keys:
                        key_html_parts.append(f"<span style=\"background-color:{color};padding:2px;border-radius:3px\">{escape(key)}</span>")
                    st.markdown("Keys: " + ", ".join(key_html_parts), unsafe_allow_html=True)

                    # highlight matches inside the child's own description as well
                    child_desc = entries[v].get('description','') or ''
                    child_matches: List[Tuple[int, int, str]] = []
                    for key in keys:
                        for (s, t) in find_key_positions(child_desc, key):
                            child_matches.append((s, t, color))
                    if child_matches:
                        st.markdown(highlight_with_colors(child_desc, child_matches), unsafe_allow_html=True)
                    else:
                        st.write(child_desc)

        # Key snipping UI collapsed by default, stays open if a snip action was just performed
        _snip_section_open = st.session_state.get("snip_open_parent") is not None
        with st.expander(f"Key snipping ({len(display_edges)} connections, {total_edges} total keys mentioned in other entries)", expanded=_snip_section_open):
            st.markdown("Select connections to snip. Connections are snipped by removing the offending key from the child entry's key list. Expand a parent to see its outgoing connections.")
            for u in [idx for idx in display_order if idx in parent_map]:
                children = parent_map[u]
                label = f"{u}. {entries[u].get('name','(no name)')} — {len(children)} target(s)"
                _parent_open = (st.session_state.get("snip_open_parent") == u)
                with st.expander(label, expanded=_parent_open):
                    st.markdown("**Targets**")

                    for uu, v, keys in children:
                        child_label = f"{v}. {entries[v].get('name','(no name)')} — {len(keys)} key(s)"
                        _child_open = (st.session_state.get("snip_open_parent") == uu and st.session_state.get("snip_open_child") == v)
                        with st.expander(child_label, expanded=_child_open):
                            # show keys with individual checkboxes
                            key_check_ids: List[str] = []
                            for i, key in enumerate(keys):
                                # larger checkbox with explicit label to be more obvious
                                kcol, kchk = st.columns([8, 2])
                                kcol.markdown(f"<span style='background-color:#eee;padding:4px;border-radius:4px'>{escape(key)}</span>", unsafe_allow_html=True)
                                chk_id = f"edge_key_{uu}_{v}_{i}"
                                key_check_ids.append(chk_id)
                                kchk.checkbox("Snip", key=chk_id)

                            if st.button("Snip selected keys for this target", key=f"snip_target_{uu}_{v}"):
                                selected = []
                                for i, key in enumerate(keys):
                                    chk_id = f"edge_key_{uu}_{v}_{i}"
                                    if st.session_state.get(chk_id, False):
                                        selected.append(key)
                                if selected:
                                    push_undo()
                                    if v not in st.session_state.snipped_keys:
                                        st.session_state.snipped_keys[v] = set()
                                    st.session_state.snipped_keys[v].update(selected)
                                    st.success(f"Snipped {len(selected)} key(s) from target {v}.")
                                    # clear the checkboxes for this target
                                    for i in range(len(keys)):
                                        chk_id = f"edge_key_{uu}_{v}_{i}"
                                        try:
                                            del st.session_state[chk_id]
                                        except Exception:
                                            pass
                                    # if the edit dialog for this target is open, update its CSV to reflect snips
                                    edit_csv_key = f"edit_keys_csv_{v}"
                                    if edit_csv_key in st.session_state:
                                        current_keys = [k.get('keyText','') for k in entries[v].get('keys', [])]
                                        sn = st.session_state.get('snipped_keys', {}).get(v, set())
                                        trig_key = f"edit_update_trigger_{v}"
                                        st.session_state[trig_key] = ", ".join([k for k in current_keys if k not in sn])
                                    st.session_state.snip_open_parent = uu
                                    st.session_state.snip_open_child = v
                                    st.rerun()
                    confirm_snip_all = st.checkbox("Confirm snip all", value=False, key=f"snip_all_confirm_{u}")
                    if st.button("Snip all from this parent- WARNING, deletes ALL keys mentioned in this entry from child key lists!!", key=f"snip_all_{u}"):
                        if not confirm_snip_all:
                            st.warning("Please confirm snip all before proceeding.")
                        else:
                            push_undo()
                            for (_, v, keys) in children:
                                if v not in st.session_state.snipped_keys:
                                    st.session_state.snipped_keys[v] = set()
                                st.session_state.snipped_keys[v].update(keys)
                            st.success(f"Snipped all {len(children)} target(s) from parent {u}.")
                            st.session_state.snip_open_parent = u
                            st.session_state.pop("snip_open_child", None)
                            st.rerun()
        # Clear snip-open tracking so expanders return to default on the next non-snip action
        st.session_state.pop("snip_open_parent", None)
        st.session_state.pop("snip_open_child", None)


if __name__ == '__main__':
    main()
