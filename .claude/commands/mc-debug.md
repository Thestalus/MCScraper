Diagnose problems with the MCScraper browser or scraper.

The user has reported: $ARGUMENTS
If blank, run a general health check.

All commands run from the repository root. Data lives in `output/` (a link to the Dropbox data folder). Use `.venv/bin/python3`, not the system Python.

Work through the following checks in order. Stop and report as soon as you find a problem — do not keep going if something is clearly broken.

**1. Process check**
Run `lsof -i :5003 | head -5` to see if the browser is running. If not, that's likely the issue — tell the user to run `make browse`.

**2. Environment and data link check**
Check that `.venv/bin/python3` exists and that `output/` resolves (`ls output/ | head`). If `output` is a broken link, the Dropbox data folder has moved or isn't synced. If `.venv` is missing, run `make install install-semantic`.

**3. Syntax check**
Run `.venv/bin/python3 -m py_compile app_factory.py browser.py services/*.py routes/*.py`. Report any syntax errors immediately.

**4. Index check**
Do NOT read the whole `output/stories_index.json` — it is over 1 GB. Read only its first 200 bytes (`head -c 200`) to confirm it starts with `{` and contains a `"stories"` key, and check its size with `wc -c`. Also check that `output/stories_meta.db` exists and is newer than the index (`ls -la`); if it is older, tell the user to run `make build-stories-db`. Show the last 3 entries of `output/update_log.json` if it exists.

**5. Chapter count check**
Run:
```
.venv/bin/python3 -c "
import sqlite3, os
db = 'output/full_text_cache.db'
if not os.path.exists(db):
    print('NO CACHE DB')
else:
    con = sqlite3.connect(db)
    rows = con.execute('SELECT id, length(full_text), (length(full_text)-length(replace(full_text,\"<!-- BREAK -->\",\"\")))/14 + 1 AS chapters FROM texts LIMIT 20').fetchall()
    for r in rows: print(r)
    con.close()
"
```
If any story shows 1 chapter but should have more, the cache is stale. Tell the user to run `make post-update` then restart the browser.

**6. Template entity check**
Grep the templates directory for HTML entity expressions missing the `| safe` filter:
```
grep -rn '&#[0-9]*;' templates/ | grep -v '| safe'
```
Any hit means that template will render raw `&#XXXX;` text instead of the symbol. Show the user the affected lines.

**7. Route registration check**
Run:
```
.venv/bin/python3 -c "
from app_factory import AppState, create_app
app = create_app(AppState())
rules = sorted(str(r) for r in app.url_map.iter_rules())
print(f'{len(rules)} routes registered')
for r in rules: print(r)
"
```
If fewer than 35 routes are registered, a blueprint likely failed to load. Check for import errors in the output.

**8. Page render check**
With the browser running, fetch a few pages and confirm each returns 200: `/`, `/liked`, `/history`, `/recommendations`, and one `/story/<id>` (take an id from `output/stories_meta.db`). For any 500, show the traceback from the browser's terminal output.

**9. Recommendations cache check**
Check if `output/recs_cache.json` exists and is non-empty. If missing, tell the user to run `make precalc-recs`.

**Summary**
After all checks, give a short bullet-point summary:
- What is healthy
- What is broken and the exact fix command to run
