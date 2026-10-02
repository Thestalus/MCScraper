Review and improve the auto_tags for a story in the MCScraper index.

The user has provided: $ARGUMENTS
This may be a story ID, a partial title, or an author name. If blank, ask the user which story they want to classify.

Steps:
1. Read `CLASSIFIER_GUIDE.md` to understand the full tagging schema and what each tag means.
2. Find the story in the index. The index is at `index.json`. Search it for the story by matching the argument against `id`, `title`, or `author` fields. If multiple matches, list them and ask the user to pick one.
3. Show the story's current `auto_tags` dict and `categories` list clearly.
4. Read a sample of the story's text if available. Check `full_text_cache.db` in the MCScraper directory using sqlite3, or look for the story's source files. Use the first ~2000 characters as a sample.
5. Review the current tags against the CLASSIFIER_GUIDE schema. For each tag field, assess:
   - Is the current value correct?
   - Are any values missing that should be set?
   - Are any values set that seem wrong?
6. Produce a clear table showing: field | current value | suggested value | reason for change (only include rows where a change is suggested).
7. Ask the user whether to apply the suggested changes. If yes, update the story's `auto_tags` in `index.json` directly using the Edit or Read+Write tools. Be careful to preserve valid JSON — read the file, find the story by id, update only its `auto_tags` field, write back.
8. Remind the user to restart the browser (`make browse`) for changes to take effect.
