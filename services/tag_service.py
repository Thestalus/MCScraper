"""
services/tag_service.py — Tag quality filtering helpers.
"""
from __future__ import annotations

# Tags where the SetFit classifier is unreliable — only display when Svengali
# community data also backs them (i.e. 'svengali' in _tag_sources[tag]).
LOW_QUALITY_SETFIT_TAGS = frozenset({
    # Pacing — BROKEN: classifier tags 88–98% of all stories
    'rapid', 'leisurely',
    # Mind control type — F1 < 0.30
    'possession',          # F1=0.288
    'body puppet',         # F1=0.289
    # Mental changes — F1 < 0.36 or dropped from AND-gate
    'induced love',        # F1=0.301
    'addiction / need',    # F1=0.304
    'corruption arc',      # F1=0.065 (dropped from AND-gate)
    'bimboization',        # F1=0.357 (classifier-only, no regex)
    'intelligence change', # F1=0.307 (classifier-only, no regex)
    # Awareness — dropped from AND-gate (F1 < 0.05)
    'denial',              # F1=0.323
    'oblivious',           # F1=0.043
    # Dominance — low precision, excluded from search filters too
    'brutal',              # F1=0.349, P=27%
    'gentle',              # F1=0.378, P=10%
    'cold / clinical',     # F1=0.048 (dropped from AND-gate)
    # Relationship — F1 < 0.41 or dropped from AND-gate
    'authority',           # F1=0.303
    'strangers',           # F1=0.405
    'reluctant subject',   # F1=0.052 (dropped from AND-gate)
    # Sexual acts — F1 < 0.30
    'creampie',            # F1=0.176 (dropped from AND-gate)
    'facial',              # F1=0.299 (scarce data)
    # Themes — dropped from AND-gate or classifier-only with low F1
    'horror / dark',       # F1=0.027 (dropped from AND-gate)
    'serial recruitment',  # F1=0.347 (classifier-only, no regex)
    'betrayal',            # F1=0.330 (classifier-only, no regex)
})


def visible_tags(tags, at):
    """Return only tags that are reliable enough to display.

    A tag passes if:
    - It is NOT in LOW_QUALITY_SETFIT_TAGS, OR
    - It IS in the set but Svengali community data backs it
      (i.e. 'svengali' in auto_tags['_tag_sources'][tag]).
    """
    if not tags:
        return []
    sources = (at.get('_tag_sources') or {}) if isinstance(at, dict) else {}
    result = []
    for tag in tags:
        if tag not in LOW_QUALITY_SETFIT_TAGS:
            result.append(tag)
        elif 'svengali' in sources.get(tag, []):
            result.append(tag)
    return result
