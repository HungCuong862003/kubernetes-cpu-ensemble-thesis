One more thing before we start coding. Update CLAUDE.md to add a "Code Style" section with these rules. Also internalize them yourself for every line of code you write in this project:

## Code Style (humanized, junior-programmer voice)

Write code as a real junior CS student would, not as an AI assistant producing polished output. Specifically:

- Comments are sparse and only where genuinely useful. Don't comment what the code obviously says ("# loop through items" above a for loop). DO comment WHY a non-obvious choice was made ("# h_steps=12 because 60min @ 5min cadence; sprint1's convention").
- No docstrings on every function. Add one only for functions whose contract isn't obvious from the signature. One-liner descriptions are fine.
- Variable names are practical, sometimes terse. `cells` not `precomputed_forecast_cells`. `df` is fine for a local pandas DataFrame. Don't over-explain.
- Section comments using `# === short label ===` banners are acceptable when a script has multiple distinct phases. Don't decorate every block.
- Print statements during development are OK. Leave a few for debugging visibility. Don't wrap everything in logging.
- No type hints unless the function is genuinely confusing without them. Streamlit page code rarely needs them.
- Imports grouped: stdlib, third-party, local. Blank line between groups. No alphabetical zealotry.
- f-strings for formatting. Don't mix .format() and f-strings.
- Error handling only where things actually fail. Don't wrap every read in try/except defensively.
- If you reach for a "clever" pattern (decorators, metaclasses, complex comprehensions), step back and use the obvious version instead. We're optimizing for a reviewer who'll read this for 30 seconds at defense.
- One blank line between functions, two between major sections. Standard PEP 8 spacing, not aggressively spaced.
- File should look like ~80% of it was written in one sitting by someone who was thinking about the problem, not refining for elegance.

What we want to avoid: emoji decorations in print statements, exhaustive type annotations, defensive try/except wrappers, three-line docstrings on five-line functions, alphabetized imports for their own sake, and the general "this was clearly produced by an LLM" feel.

What we want to keep: working code, comments where they earn their space, occasional inline TODO or FIXME if you genuinely punt on something.

After updating CLAUDE.md, confirm you've internalized this, then we proceed with Piece 1.