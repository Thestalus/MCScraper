Diagnose problems with the MCScraper browser or scraper.

The user has reported: $ARGUMENTS
If blank, run a general health check.

Work through the following checks in order. Stop and report as soon as you find a problem — do not keep going if something is clearly broken.

**1. Process check**
Run `lsof -i :5000 | head -5` to see if the browser is running. If not, that's likely the issue — tell the user to run `make browse`.

**2. Syntax check**
Run `python3 -m py_compile app_factory.py` and the files in `routes/`. Report any syntax errors immediately.

**3. Index check**
Read the first 200 bytes of `index.json` to confirm it is valid JSON (starts with `[`). Check `wc -c index.json` for file size — if it is 0 or very small the index is empty. Also check if `update_log.json` exists and show the last 3 entries.

**4. Chapter count check**
Run:
```
python3 -c "
import sqlite3, os
db = 'full_text_cache.db'
if not os.path.exists(db):
    print('NO CACHE DB')
else:
    con = sqlite3.connect(db)
    rows = con.execute('SELECT id, length(text), (length(text)-length(replace(text,\"<!-- BREAK -->\",\"\")))//16 + 1 as chapters FROM cache LIMIT 20').fetchall()
    for r in rows: print(r)
    con.close()
"
```
If any story shows 1 chapter but should have more, the cache is stale. Tell the user to run `make post-update` then restart the browser.

**5. Template entity check**
Grep the templates directory for HTML entity expressions missing the `| safe` filter:
```
grep -n '&#[0-9]*;' templates/*.html | grep -v '| safe'
```
Any hit means that template will render raw `&#XXXX;` text instead of the symbol. Show the user the affected lines.

**6. Route registration check**
Run:
```
python3 -c "
import sys; sys.path.insert(0, '.')
from app_factory import create_app
from data_loader import load_data
from app_factory import AppState
state = AppState()
load_data(state)
app = create_app(state)
rules = sorted(str(r) for r in app.url_map.iter_rules())
print(f'{len(rules)} routes registered')
for r in rules: print(r)
"
```
If fewer than 30 routes are registered, a blueprint likely failed to load. Check for import errors in the output.

**7. Recommendations cache check**
Check if `recs_cache.json` exists and is non-empty. If missing, tell the user to run `make precalc-recs`.

**Summary**
After all checks, give a short bullet-point summary:
- What is healthy
- What is broken and the exact fix command to run
