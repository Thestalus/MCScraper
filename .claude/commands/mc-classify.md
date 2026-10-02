Review and improve the auto_tags for a story in the MCScraper index.

The user has provided: $ARGUMENTS
This may be a story ID, a partial title, or an author name. If blank, ask the user which story they want to classify.

All commands run from the repository root. Data lives in `output/` (a link to the Dropbox data folder). Use `.venv/bin/python3`, not the system Python.

Steps:
1. Read `CLASSIFIER_GUIDE.md` to understand the full tagging schema and what each tag means.
2. Find the story. Do NOT open `output/stories_index.json` with Read/Edit — it is over 1 GB. Search the slim metadata DB instead:
   ```
   .venv/bin/python3 -c "
   import sqlite3, sys
   q = sys.argv[1].lower()
   con = sqlite3.connect('output/stories_meta.db')
   for r in con.execute('SELECT id, title, author FROM stories WHERE lower(id) LIKE ? OR lower(title) LIKE ? OR lower(author) LIKE ? LIMIT 20', (f'%{q}%',)*3):
       print(r)
   " "<argument>"
   ```
   If multiple matches, list them and ask the user to pick one.
3. Show the story's current `auto_tags` and `categories` clearly:
   `SELECT auto_tags, categories FROM stories WHERE id = ?` on `output/stories_meta.db` (both are JSON strings).
4. Read a sample of the story's text: `SELECT substr(full_text, 1, 2000) FROM texts WHERE id = ?` on `output/full_text_cache.db`.
5. Review the current tags against the CLASSIFIER_GUIDE schema. For each tag field, assess:
   - Is the current value correct?
   - Are any values missing that should be set?
   - Are any values set that seem wrong?
6. Produce a clear table showing: field | current value | suggested value | reason for change (only include rows where a change is suggested).
7. Ask the user whether to apply the suggested changes. If yes, make sure the browser and any tagging/update jobs are stopped, then update `output/stories_index.json` with a Python script: load the JSON, find the story in `data["stories"]` by `id`, change only its `auto_tags` field, write to a temporary file in `output/`, then rename it over the original.
8. Rebuild the derived databases so the browser sees the change: `make build-stories-db && make build-search-db`. Then remind the user to restart the browser (`make browse`).
