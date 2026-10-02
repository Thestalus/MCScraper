"""
routes/generate.py — /generate, /api/llm/models, /api/generate/retrieve, /api/generate/stream
"""
from __future__ import annotations
from pathlib import Path

from flask import Blueprint, render_template, request, jsonify, current_app
from flask import Response, stream_with_context

from services.search_service import search_stories
from services.data_store import get_auto_tags

generate_bp = Blueprint("generate", __name__)


@generate_bp.route("/generate")
def generate_page():
    from tagger import (SETTING_RULES, THEME_RULES, TONE_RULES,
                        MIND_CONTROL_TYPE_RULES, DOMINANCE_RULES, PACING_RULES,
                        AWARENESS_RULES)
    mc_types       = [r[0] for r in MIND_CONTROL_TYPE_RULES]
    settings       = [r[0] for r in SETTING_RULES]
    themes         = [r[0] for r in THEME_RULES]
    tones          = [r[0] for r in TONE_RULES]
    dominances     = [r[0] for r in DOMINANCE_RULES]
    pacings        = [r[0] for r in PACING_RULES]
    awareness_opts = [r[0] for r in AWARENESS_RULES]
    return render_template(
        "generate.html",
        mc_types=mc_types, settings=settings, themes=themes,
        tones=tones, dominances=dominances, pacings=pacings,
        awareness_opts=awareness_opts,
        max_tokens=3000,
    )


@generate_bp.route("/api/llm/models", methods=["POST"])
def api_llm_models():
    """Check connection and list available models from the local LLM."""
    from generator import LLMClient
    data   = request.get_json(force=True)
    url    = data.get("url", "http://localhost:1234/v1")
    key    = data.get("key", "none")
    client = LLMClient(base_url=url, api_key=key)
    models = client.list_models()
    return jsonify({"models": models, "connected": len(models) > 0})


@generate_bp.route("/api/generate/retrieve", methods=["POST"])
def api_generate_retrieve():
    """RAG: find similar stories to use as generation examples."""
    state = current_app.state
    data  = request.get_json(force=True)
    query = data.get("query", "")
    tags  = data.get("tags", {})

    full_query = " ".join(filter(None, [
        query,
        tags.get("mc_type", ""),
        tags.get("setting", ""),
        tags.get("tone", ""),
        " ".join(tags.get("themes", [])),
    ]))

    examples = []
    if state.rec and state.engine._loaded and full_query.strip():
        if state.engine.semantic:
            import numpy as np
            qvec = state.engine._model.encode([full_query], normalize_embeddings=True)[0]
            scores = []
            for sid, s in state.rec.stories.items():
                sv = state.engine._embeddings.get(sid)
                if sv is None:
                    continue
                score = float(np.dot(sv, qvec))
                scores.append((s, score))
            scores.sort(key=lambda x: -x[1])
            examples = [
                {
                    "id":          s["id"],
                    "title":       s.get("title", ""),
                    "author":      s.get("author", ""),
                    "score":       round(sc, 3),
                    "categories":  s.get("categories", []),
                    "auto_tags":   get_auto_tags(s),
                    "description": s.get("description", ""),
                    "full_text":   s.get("full_text", ""),
                    "word_count":  s.get("word_count", 0),
                }
                for s, sc in scores[:5]
            ]
        else:
            results, _ = search_stories(
                state, query=query,
                mc_type=tags.get("mc_type", ""),
                setting=tags.get("setting", ""),
                tone=tags.get("tone", ""),
                per_page=5,
            )
            examples = [
                {
                    "id":          s["id"],
                    "title":       s.get("title", ""),
                    "author":      s.get("author", ""),
                    "score":       0.5,
                    "categories":  s.get("categories", []),
                    "auto_tags":   get_auto_tags(s),
                    "description": s.get("description", ""),
                    "full_text":   s.get("full_text", ""),
                    "word_count":  s.get("word_count", 0),
                }
                for s in results[:5]
            ]
    elif full_query.strip():
        results, _ = search_stories(state, query=query, per_page=5)
        examples = [
            {
                "id":          s["id"],
                "title":       s.get("title", ""),
                "author":      s.get("author", ""),
                "score":       0.5,
                "categories":  s.get("categories", []),
                "auto_tags":   get_auto_tags(s),
                "description": s.get("description", ""),
                "full_text":   s.get("full_text", ""),
                "word_count":  s.get("word_count", 0),
            }
            for s in results[:5]
        ]

    display_examples = [
        {"id": e["id"], "title": e["title"], "author": e["author"], "score": e["score"]}
        for e in examples
    ]
    return jsonify({"examples": display_examples})


@generate_bp.route("/api/generate/stream", methods=["POST"])
def api_generate_stream():
    """Stream story generation from local LLM."""
    from generator import LLMClient, build_prompt, estimate_tokens
    state = current_app.state

    data         = request.get_json(force=True)
    request_text = data.get("request_text", "")
    example_ids  = [e["id"] for e in data.get("examples", [])]
    tags         = data.get("tags", {})
    mode         = data.get("mode", "inspired")
    url          = data.get("url", "http://localhost:1234/v1")
    key          = data.get("key", "none")
    model_name   = data.get("model", "auto")
    temperature  = float(data.get("temperature", 0.85))
    max_tokens   = int(data.get("max_tokens", 3000))

    examples = []
    for eid in example_ids:
        s = state.data.get("by_id", {}).get(eid)
        if s:
            examples.append(s)
    if not examples and request_text:
        results, _ = search_stories(state, query=request_text, per_page=3)
        examples = results

    length_class = tags.get("length", "short story")
    max_tokens   = estimate_tokens(length_class)

    client   = LLMClient(base_url=url, api_key=key, model=model_name)
    messages = build_prompt(request_text, examples, tags, style_mode=mode)

    if not client.ping():
        return jsonify({
            "error": "Cannot connect to LLM. Make sure LM Studio or Ollama is running."
        }), 503

    def generate_stream():
        try:
            for chunk in client.complete(
                messages, max_tokens=max_tokens, temperature=temperature, stream=True
            ):
                yield chunk
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
