# Lorebook Converter Tool

A Streamlit app for converting and cleaning roleplay lorebooks between SillyTavern and DreamJourney formats. Upload or paste JSON, reorder entries, inspect cascade triggers, snip unwanted key links, and export in your preferred format.

## Features
- Upload or paste lorebook JSON
- Auto-detects SillyTavern vs DreamJourney (with override)
- Reorder entries (move up/down, move to position, bulk reorder)
- Cascade cleanup: inspect trigger edges and highlights
- Snip keys per target or snip all from a parent
- Quick edit entry description and keys
- Export as SillyTavern or DreamJourney
- Reset snips/edits/order and reset all

## Requirements
- Python 3.9+

## Setup
1. Create a virtual environment (recommended).
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run
From the project folder:
- `streamlit run app.py`

## Samples
See the sample files in [samples/](samples/).

## Notes
- Exports are generated from the current working state (reorders, snips, edits).
- Use “New Import” to clear state between different source files.
