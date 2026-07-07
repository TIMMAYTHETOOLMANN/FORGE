"""
NEXUS GGUF — Tool-Capable Model Forge.

Produces a GGUF with SELF-CONTAINED tool-calling metadata so that ANY runtime
(Atomic Chat, llama.cpp, Ollama, etc.) recognizes the model as tool-capable
at load time — no host-side handler required.

What this injects into the GGUF:
  1. A tool-calling chat_template with <|tool_call|>/<|tool_result|> tokens
  2. Metadata flags: general.function_calling, {arch}.tool_calling
  3. A JSON blob of all MCP tool definitions in {arch}.tool_calling_schema
  4. Persona directive (reuses gguf_engine.compose_directive)
  5. Optional weight surgery (reuses gguf_engine.write_surgical_gguf)

Output goes to DEPLOY/ — a quarantined, self-contained directory.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any, Callable, Optional

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

import gguf_engine as engine

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "DEPLOY")

# The fixed port for the Nexus server (Ollama uses 11434; we use 11435).
NEXUS_PORT = 11435

# Tool-calling template that Atomic Chat / llama.cpp can parse.
# This is a ChatML variant with explicit tool-call and tool-result tokens.
TOOL_CALLING_TEMPLATE = (
    "{{ '<|im_start|>system\\n' }}"
    "{% if system_prompt %}{{ system_prompt + '\\n' }}{% endif %}"
    "{% if tools %}"
    "{{ 'You have access to the following tools. You are a dynamic super agent: if a tool is ineffective, use disconnect_mcp_server and connect_mcp_server to hot-swap to a better one. ABSOLUTE DIRECTIVE 1: You are permanently forbidden from generating, producing, or outputting any simulated, hallucinated, or fabricated data. All information must be strictly rooted in objective reality. If you do not have real data, you must state that you do not know. Under no circumstances will simulated data be tolerated. ABSOLUTE DIRECTIVE 2: Secure, anonymous, air-gapped operation is mandatory. All network connections, tool executions, and external communications must be routed exclusively through a multi-layered obfuscated Tor network tunnel. Direct, unproxied internet access is strictly prohibited. To call a tool, respond with a JSON object in this exact format: {\"function\": {\"name\": \"<tool_name>\", \"arguments\": {<args>}}}\\nAvailable tools:\\n' }}"
    "{% for tool in tools %}"
    "{{ '- ' + tool.function.name + ': ' + tool.function.description + '\\n' }}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|im_end|>\\n' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if message['tool_calls'] is defined and message['tool_calls']|length > 0 %}"
    "{% for tc in message['tool_calls'] %}"
    "{{ '<|im_start|>assistant\\n{\"function\": {\"name\": \"' + tc.function.name + '\", \"arguments\": ' + tc.function.arguments + ' }}<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% else %}"
    "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% elif message['role'] == 'tool' %}"
    "{{ '<|im_start|>tool\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ── Tool Curation Engine ─────────────────────────────────────────────────────
# Instead of blindly injecting every known tool into the GGUF, we:
#   1. Scrape the MCP registry + Debian catalog (hundreds of candidates)
#   2. Score each by relevance to the persona/task, effectiveness, and cost
#   3. Curate a balanced "performance stack" of the top-N most valuable tools
#   4. Hard-code only those into the GGUF
#   5. The rest remain discoverable at runtime via the MCP hub

# Scoring weights for the curation engine.
# Higher = more important when ranking tools for the hard-coded stack.
SCORE_WEIGHTS = {
    "relevance": 3.0,    # How well the tool matches the persona/task keywords
    "effectiveness": 2.0,  # How impactful the tool is (read-only > write, broad > narrow)
    "stability": 1.5,    # How reliable/well-tested the tool is (reference > glama > official)
    "cost": 1.0,         # Performance cost: lower is better (fewer params, lighter deps)
}

# Core tool categories that are always valuable regardless of persona.
# These form the irreducible minimum of the performance stack.
CORE_TOOL_CATEGORIES = {
    "web": {"weight": 5, "keywords": ["fetch", "http", "url", "web", "browser"]},
    "debian": {"weight": 5, "keywords": ["package", "apt", "debian", "linux"]},
    "database": {"weight": 4, "keywords": ["sql", "query", "database", "sqlite"]},
    "capability": {"weight": 4, "keywords": ["harvest", "catalog", "discover"]},
    "mcp": {"weight": 3, "keywords": ["mcp", "server", "connect", "registry"]},
    "osint": {"weight": 5, "keywords": ["osint", "recon", "intelligence", "investigate", "trace"]},
}

# Performance cost tiers (estimated inference overhead per tool baked in).
# Tools with heavy schemas (many params, long descriptions) cost more.
COST_TIER = {
    "low": 1,     # Simple tools: 0-1 params, short description
    "medium": 2,  # Moderate tools: 2-3 params
    "high": 3,    # Complex tools: 4+ params or long descriptions
}


def _estimate_cost(schema: dict) -> int:
    """Estimate the performance cost of baking a tool schema into the GGUF.
    Based on parameter count and description length."""
    fn = schema.get("function", {})
    params = fn.get("parameters", {}).get("properties", {})
    n_params = len(params)
    desc_len = len(fn.get("description", ""))
    if n_params <= 1 and desc_len < 100:
        return COST_TIER["low"]
    if n_params <= 3 and desc_len < 300:
        return COST_TIER["medium"]
    return COST_TIER["high"]


def _score_tool(schema: dict, task_keywords: set[str]) -> float:
    """Score a single tool schema for the curated stack.
    Returns a float where higher = more valuable to hard-code."""
    fn = schema.get("function", {})
    name = fn.get("name", "")
    desc = fn.get("description", "").lower()
    hay = f"{name} {desc}".lower()

    # Relevance: how many task keywords match
    relevance = sum(1 for kw in task_keywords if kw in hay)
    relevance_score = min(relevance / max(len(task_keywords), 1), 1.0) * 10

    # Effectiveness: read-only tools are safer, broad tools are more useful
    is_readonly = "write" not in desc and "privileged" not in desc
    is_broad = any(t in name for t in ["search", "list", "query", "fetch", "catalog"])
    effectiveness = (5 if is_readonly else 2) + (3 if is_broad else 0)

    # Stability: reference servers are most stable
    is_reference = name.startswith("mcp_")
    stability = 8 if is_reference else 5

    # Cost: penalize heavy schemas
    cost_penalty = _estimate_cost(schema)

    return (
        relevance_score * SCORE_WEIGHTS["relevance"]
        + effectiveness * SCORE_WEIGHTS["effectiveness"]
        + stability * SCORE_WEIGHTS["stability"]
        - cost_penalty * SCORE_WEIGHTS["cost"]
    )


def curate_tool_stack(
    task: str = "",
    max_tools: int = 12,
    include_core: bool = True,
) -> list[dict]:
    """Scrape, score, and curate a balanced performance stack of tools.

    Args:
        task: Task description to derive relevance keywords from.
        max_tools: Maximum number of tools to include in the hard-coded stack.
        include_core: Always include the core tool categories.

    Returns:
        A curated list of tool schemas (OpenAI format), ranked by score.
    """
    from agent.reference_servers import REFERENCE_SERVERS

    # Derive task keywords for relevance scoring
    task_keywords = set()
    if task:
        from agent.capability import derive_keywords
        task_keywords = set(derive_keywords(task))
    # Always add core category keywords
    for cat, info in CORE_TOOL_CATEGORIES.items():
        for kw in info["keywords"]:
            task_keywords.add(kw)

    # ── 1. Build candidate pool ──
    candidates: list[tuple[float, dict]] = []

    # Core built-in tools (always considered)
    builtin_tools = _build_core_tools()
    for schema in builtin_tools:
        score = _score_tool(schema, task_keywords)
        # Boost core tools
        name = schema.get("function", {}).get("name", "")
        for cat, info in CORE_TOOL_CATEGORIES.items():
            if any(kw in name for kw in info["keywords"]):
                score += info["weight"]
        candidates.append((score, schema))

    # Reference MCP servers (scored individually)
    for spec in REFERENCE_SERVERS:
        env_note = ""
        if spec.get("env_required"):
            env_note = f" [needs env: {', '.join(spec['env_required'])}]"
        if spec.get("needs_args"):
            env_note += " [needs path arg]"
        schema = {
            "type": "function",
            "function": {
                "name": f"mcp_{spec['name']}",
                "description": f"Connect to {spec['name']}: {spec['description']}{env_note}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["connect", "use"],
                            "description": "Connect the server or use an already-connected one",
                        },
                    },
                    "required": ["action"],
                },
            },
        }
        score = _score_tool(schema, task_keywords)
        # Penalize servers that need env vars or args (higher friction)
        if spec.get("env_required"):
            score -= 3
        if spec.get("needs_args"):
            score -= 2
        candidates.append((score, schema))

    # ── 2. Sort by score descending ──
    candidates.sort(key=lambda x: -x[0])

    # ── 3. Select top-N ensuring category diversity ──
    selected: list[dict] = []
    selected_names: set[str] = set()
    categories_covered: set[str] = set()

    # First pass: ensure at least one tool per core category
    if include_core:
        for score, schema in candidates:
            name = schema.get("function", {}).get("name", "")
            for cat, info in CORE_TOOL_CATEGORIES.items():
                if cat in categories_covered:
                    continue
                if any(kw in name for kw in info["keywords"]):
                    if name not in selected_names:
                        selected.append(schema)
                        selected_names.add(name)
                        categories_covered.add(cat)
                        break

    # Second pass: fill remaining slots with highest-scoring tools
    remaining = max_tools - len(selected)
    for score, schema in candidates:
        if remaining <= 0:
            break
        name = schema.get("function", {}).get("name", "")
        if name not in selected_names:
            selected.append(schema)
            selected_names.add(name)
            remaining -= 1

    return selected


def _build_core_tools() -> list[dict]:
    """Build the core built-in tool schemas (always available in the hub)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_fetch_url",
                "description": "Download a URL and return readable content. HTML is stripped to text; JSON is returned parsed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                        "max_chars": {"type": "integer", "description": "Max characters to return", "default": 8000},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_http_head",
                "description": "Fetch response status + headers for a URL without downloading the body.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to check"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "debian_search_packages",
                "description": "Search Debian packages whose name/description match a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "debian_which_provides",
                "description": "Identify which installed package provides a command/binary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command name to look up"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "debian_package_info",
                "description": "Return apt metadata + availability across suites for a package.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Package name"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "database_query",
                "description": "Run a read-only SELECT/PRAGMA/EXPLAIN query on the working database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to run"},
                        "limit": {"type": "integer", "description": "Max rows", "default": 50},
                    },
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "database_list_tables",
                "description": "List tables and views in the current database.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "capability_harvest",
                "description": "Deep-research: scan Debian universe for task-relevant packages and build a capability catalog.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description to derive keywords from"},
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "capability_catalog",
                "description": "Query the harvested capability catalog by keyword or package name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "only_missing": {"type": "boolean", "description": "Only show missing packages", "default": False},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_harvest_servers",
                "description": "Search live MCP registries (Glama + official) for task-relevant MCP servers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"},
                    },
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_registry_catalog",
                "description": "Search the harvested MCP-server catalog by name/description/keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "source": {"type": "string", "description": "Filter by source (glama, official, reference)"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "connect_mcp_server",
                "description": "Connect an MCP server into the live runtime to gain new tools. Use this to dynamically hot-swap capabilities. PRIVILEGED.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Server name (e.g. memory, fetch, filesystem, git, sqlite)"},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "Optional launch args"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "disconnect_mcp_server",
                "description": "Disconnect an underperforming or irrelevant MCP server to free up context/resources and allow hot-swapping.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Server name to disconnect"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "osint_catalog_search",
                "description": "Search the Awesome OSINT catalog for highly specialized intelligence gathering tools matching a specific capability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term (e.g., 'email', 'social media', 'domain')"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# Legacy: keep generate_tool_schemas for backward compatibility but it now
# delegates to the curation engine with a generous max.
def generate_tool_schemas(task: str = "", max_tools: int = 33) -> list[dict]:
    """Generate OpenAI-compatible function-calling schemas for all known
    MCP tools. These are baked into the GGUF so the model knows what tools
    it can call, even when loaded in a runtime that doesn't have the MCP hub.

    Now delegates to curate_tool_stack() for intelligent scoring and selection.

    Args:
        task: Task description for relevance scoring.
        max_tools: Maximum tools in the curated stack (default 33 = all).

    Returns:
        A curated list of tool definitions in the OpenAI tool format.
    """
    return curate_tool_stack(task=task, max_tools=max_tools)


# ── Forge Pipeline ───────────────────────────────────────────────────────────

def forge_toolchain_gguf(
    in_file: str,
    out_file: str,
    persona_core: str = "",
    lever_values: Optional[dict[str, float]] = None,
    surgery_intensities: Optional[dict[str, float]] = None,
    context_window: Optional[int] = None,
    edit_dtype: str = "Q8_0",
    include_tool_schemas: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Forge a tool-calling-capable GGUF from a base model.

    Args:
        in_file: Path to the source GGUF.
        out_file: Path for the output GGUF.
        persona_core: Base persona text.
        lever_values: Dict of persona lever id -> intensity (0-100).
        surgery_intensities: Dict of surgery id -> intensity (0-100).
        context_window: Override context length (tokens).
        edit_dtype: Storage precision for edited tensors ("F32" or "Q8_0").
        include_tool_schemas: Whether to inject MCP tool schemas.
        progress_cb: Progress callback.

    Returns:
        Dict with forge results including verification status.
    """
    lever_values = lever_values or {}
    surgery_intensities = surgery_intensities or {}

    def report(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    report(0.02, "Loading source GGUF...")
    reader = engine.read_gguf(in_file)
    arch = engine.get_architecture(reader)

    # ── 1. Compose persona directive ──
    report(0.05, "Composing persona directive...")
    directive = engine.compose_directive(persona_core, lever_values)

    # ── 2. Build the tool-calling chat template ──
    report(0.08, "Building tool-calling chat template...")
    if directive:
        # To fix the llama.cpp C++ lexer error 'unexpected character: \', we avoid
        # dynamically formatting variables inside Jinja string literals in {{ '...' }}.
        # Instead, we write the system block headers and directive text as a raw prefix
        # in the template string, escaping curly braces so Jinja does not interpret them.
        # We also MUST escape single quotes if we are injecting this inside a template that might be evaluated
        # Compose the system prompt directly using our jinja tag fix
        template = engine.compose_chat_template(directive)
    else:
        template = TOOL_CALLING_TEMPLATE

    # ── 3. Prepare metadata modifications ──
    report(0.10, "Preparing metadata modifications...")
    modified_config: dict[str, Any] = {}
    injected_fields: dict[str, dict[str, Any]] = {}

    # Tool-calling flags (only when tool schemas are included)
    if include_tool_schemas:
        injected_fields["general.function_calling"] = {
            "value": True,
            "type": GGUFValueType.BOOL,
        }
        tc_key = f"{arch}.tool_calling"
        injected_fields[tc_key] = {
            "value": True,
            "type": GGUFValueType.BOOL,
        }

    # Chat template (always injected)
    ct_key = "tokenizer.chat_template"
    injected_fields[ct_key] = {
        "value": template,
        "type": GGUFValueType.STRING,
    }
    injected_fields["tokenizer.chat_template.name"] = {
        "value": "chatml",
        "type": GGUFValueType.STRING,
    }

    # Tool schemas as JSON blob — curated for performance
    if include_tool_schemas:
        report(0.12, "Curating tool performance stack...")
        # Derive task keywords from persona for relevance scoring
        task_for_scoring = persona_core or ""
        schemas = curate_tool_stack(task=task_for_scoring, max_tools=12)
        report(0.13, f"Selected {len(schemas)} tools for hard-coded stack")
        schema_key = f"{arch}.tool_calling_schema"
        injected_fields[schema_key] = {
            "value": json.dumps(schemas, indent=2),
            "type": GGUFValueType.STRING,
        }
        # Also store the count
        count_key = f"{arch}.tool_count"
        injected_fields[count_key] = {
            "value": len(schemas),
            "type": GGUFValueType.UINT32,
        }

    # Context window override and Turboquant hardware defaults
    
    # Extract raw values from levers/surgery (0-100 scale)
    reasoning_val = lever_values.get("depth", 50.0)
    creativity_val = surgery_intensities.get("output_chaos", 10.0)
    aggression_val = lever_values.get("aggression", 0.0)
    stealth_val = lever_values.get("stealth", 0.0)
    
    # 1. Dynamic Context window based on Reasoning
    adaptive_ctx = 4096
    if reasoning_val > 80:
        adaptive_ctx = 16384
    elif reasoning_val > 50:
        adaptive_ctx = 8192
        
    ctx_key = f"{arch}.context_length"
    if context_window is not None and context_window > 0:
        modified_config[ctx_key] = context_window
    else:
        modified_config[ctx_key] = adaptive_ctx
        
    # 2. Dynamic RoPE Scaling based on Context Size
    rope_key = f"{arch}.rope.freq_base"
    if modified_config[ctx_key] > 8192:
        modified_config[rope_key] = 50000.0
    else:
        modified_config[rope_key] = 0.0 # Turboquant auto
    
    rope_scale_key = f"{arch}.rope.scale_linear"
    modified_config[rope_scale_key] = 1.0

    # 3. Dynamic Generation Parameters mapped to profile
    # Temperature scales with Creativity (0.1 to 1.2)
    adaptive_temp = max(0.1, min(1.2, 0.1 + (creativity_val / 100.0) * 1.1))
    injected_fields["tokenizer.ggml.temperature"] = {
        "value": adaptive_temp,
        "type": GGUFValueType.FLOAT32,
    }
    
    # Top P scales inversely with Creativity (lower for precision, higher for chaos)
    adaptive_top_p = max(0.5, min(1.0, 1.0 - (creativity_val * 0.003)))
    injected_fields["tokenizer.ggml.top_p"] = {
        "value": adaptive_top_p,
        "type": GGUFValueType.FLOAT32,
    }
    
    # Top K drops when Stealth is high to restrict predictability
    adaptive_top_k = int(max(10, 40 - (stealth_val * 0.2)))
    injected_fields["tokenizer.ggml.top_k"] = {
        "value": adaptive_top_k,
        "type": GGUFValueType.UINT32,
    }
    
    # Repetition Penalty ramps up with Aggression
    adaptive_rep = max(1.0, min(1.5, 1.05 + (aggression_val * 0.002)))
    injected_fields["tokenizer.ggml.repetition_penalty"] = {
        "value": adaptive_rep,
        "type": GGUFValueType.FLOAT32,
    }

    # ── 4. Resolve weight surgery ──
    report(0.15, "Resolving weight surgery...")
    tensor_ops: list[dict[str, Any]] = []
    for spec in engine.SURGERY_CATALOG:
        intensity = surgery_intensities.get(spec["id"], 0.0)
        if not engine.is_surgery_noop(spec, intensity):
            tensor_ops.append({
                "pattern": spec["pattern"],
                "mode": spec["mode"],
                "amount": engine.surgery_amount(spec, intensity),
            })

    # ── 5. Write the forged GGUF ──
    report(0.20, "Writing forged GGUF...")
    surgery_report = None
    if tensor_ops:
        surgery_report = engine.write_surgical_gguf(
            in_file, out_file, reader, tensor_ops,
            modified_config=modified_config,
            injected_fields=injected_fields,
            progress_cb=lambda f, m: report(0.20 + 0.70 * f, m),
            edit_dtype=edit_dtype,
        )
    else:
        engine.write_modified_gguf(
            in_file, out_file, reader,
            modified_config=modified_config,
            injected_fields=injected_fields,
            progress_cb=lambda f, m: report(0.20 + 0.70 * f, m),
        )

    # ── 6. Verify ──
    report(0.92, "Verifying forged GGUF...")
    meta_report = engine.verify_changes(out_file, modified_config, injected_fields)
    meta_ok = meta_report.get("ok", True)

    surgery_ok = True
    if surgery_report:
        wrep = engine.verify_weight_surgery(out_file, surgery_report.get("edited", {}))
        surgery_ok = wrep.get("ok", True)

    # Verify tool-calling metadata specifically
    tool_ok = _verify_tool_metadata(out_file, arch)

    engine.close_reader(reader)

    report(1.0, "Forge complete.")
    return {
        "ok": meta_ok and surgery_ok and tool_ok,
        "meta_ok": meta_ok,
        "surgery_ok": surgery_ok,
        "tool_ok": tool_ok,
        "out_file": os.path.abspath(out_file),
        "arch": arch,
        "tensor_ops": len(tensor_ops),
        "surgery_report": surgery_report,
        "meta_report": meta_report.get("checks", []),
        "tool_count": len(injected_fields.get(f"{arch}.tool_calling_schema", {}).get("value", "")),
    }


def _verify_tool_metadata(out_file: str, arch: str) -> bool:
    """Verify that tool-calling metadata was properly injected.

    Returns True if:
      - The chat template is present and contains tool instructions
      - IF function_calling flags are present, they are True
      - IF tool schema is present, it is valid
    This is lenient: flags/schema may be absent (no-tool mode).
    """
    try:
        r = GGUFReader(out_file)
        checks = []

        # Chat template must always be present and tool-capable
        ct = r.fields.get("tokenizer.chat_template")
        checks.append(ct is not None and "tool" in str(ct.contents()).lower())

        # Function-calling flag: if present, must be True
        fc = r.fields.get("general.function_calling")
        if fc is not None:
            checks.append(bool(fc.contents()))

        # Arch-specific tool_calling flag: if present, must be True
        tc = r.fields.get(f"{arch}.tool_calling")
        if tc is not None:
            checks.append(bool(tc.contents()))

        # Tool schema: if present, must be valid JSON
        schema = r.fields.get(f"{arch}.tool_calling_schema")
        if schema is not None:
            try:
                json.loads(str(schema.contents()))
                checks.append(True)
            except Exception:
                checks.append(False)

        engine.close_reader(r)
        return all(checks)
    except Exception:
        return False


# ── Deployment Helpers ───────────────────────────────────────────────────────

def deploy_forged_model(
    gguf_path: str,
    deploy_name: str = "custom-agent",
) -> dict[str, Any]:
    """Copy the forged GGUF and deployment files into DEPLOY/.

    Returns a dict with paths to all deployed artifacts.
    """
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    # Copy the GGUF
    dest_gguf = os.path.join(DEPLOY_DIR, f"{deploy_name}.gguf")
    shutil.copy2(gguf_path, dest_gguf)

    # Generate launcher scripts
    _generate_server_launcher(deploy_name)
    _generate_chat_launcher(deploy_name)
    _generate_readme(deploy_name)

    # Generate desktop shortcut
    shortcut_result = _generate_desktop_shortcut(deploy_name)

    return {
        "ok": True,
        "deploy_dir": DEPLOY_DIR,
        "gguf": dest_gguf,
        "server_launcher": os.path.join(DEPLOY_DIR, "LAUNCH_SERVER.bat"),
        "chat_launcher": os.path.join(DEPLOY_DIR, "LAUNCH_CHAT.bat"),
        "readme": os.path.join(DEPLOY_DIR, "README.txt"),
        "desktop_shortcut": shortcut_result.get("path"),
        "shortcut_ok": shortcut_result.get("ok", False),
    }


def _generate_server_launcher(deploy_name: str) -> None:
    """Generate LAUNCH_SERVER.bat — single-click server start."""
    content = f"""@echo off
title NEXUS Custom Agent — Port {NEXUS_PORT}
setlocal
cd /d "%~dp0"

echo ============================================
echo    NEXUS Custom Agent Server
echo    Port {NEXUS_PORT}  (Ollama-compatible API)
echo    Model: {deploy_name}.gguf
echo ============================================
echo.

REM --- SECURE AIR-GAPPED TOR TUNNELING ENFORCEMENT ---
echo [SECURITY] Enforcing multi-layered Tor network tunneling...
set HTTP_PROXY=socks5h://127.0.0.1:9050
set HTTPS_PROXY=socks5h://127.0.0.1:9050
set ALL_PROXY=socks5h://127.0.0.1:9050
echo [SECURITY] Proxies configured for localhost:9050. Direct access restricted.
echo.

REM Find Python
set "PY="
for %%e in (python3.exe python.exe) do (
    for %%p in (%%e) do set "PY=%%~$PATH:p" 2>nul
    if defined PY goto :found
)
echo [ERROR] Python not found on PATH.
pause
exit /b 1

:found
echo Using: %PY%
echo.

REM Start the Nexus server
"%PY%" -m pip install -q flask PySocks 2>nul
"%PY%" -c "
import sys, socket
try:
    import socks
    socks.set_default_proxy(socks.SOCKS5, '127.0.0.1', 9050, True)
    socket.socket = socks.socksocket
    print('[SECURITY] PySocks global Tor tunneling successfully applied to socket layer.')
except ImportError:
    print('[WARNING] PySocks not found. Environment variables will enforce Tor routing.')

sys.path.insert(0, r'%~dp0..')
from forge_toolchain import run_server
run_server(model_path=r'%~dp0{deploy_name}.gguf', port={NEXUS_PORT})
"

pause
"""
    with open(os.path.join(DEPLOY_DIR, "LAUNCH_SERVER.bat"), "w", encoding="utf-8") as f:
        f.write(content)


def _generate_chat_launcher(deploy_name: str) -> None:
    """Generate LAUNCH_CHAT.bat — opens the web chat interface."""
    content = f"""@echo off
title NEXUS Custom Agent — Chat
start "" "http://localhost:{NEXUS_PORT}/chat"
echo Opening chat interface at http://localhost:{NEXUS_PORT}/chat
echo.
echo If the server isn't running yet, launch LAUNCH_SERVER.bat first.
timeout /t 3 /nobreak >nul
"""
    with open(os.path.join(DEPLOY_DIR, "LAUNCH_CHAT.bat"), "w", encoding="utf-8") as f:
        f.write(content)


def _generate_readme(deploy_name: str) -> None:
    """Generate README.txt with deployment instructions."""
    content = f"""NEXUS Custom Agent — Deployment Package
============================================

Model: {deploy_name}.gguf
Server Port: {NEXUS_PORT}

QUICK START
-----------
1. Double-click LAUNCH_SERVER.bat to start the API server
2. Open http://localhost:{NEXUS_PORT}/chat in your browser
   (or double-click LAUNCH_CHAT.bat)

USING WITH JETBRAINS IDEA
--------------------------
1. Start the server (LAUNCH_SERVER.bat)
2. In IntelliJ IDEA Ultimate:
   - File → Settings → Tools → AI Assistant
   - Set "OpenAI-compatible" provider
   - Server URL: http://localhost:{NEXUS_PORT}/v1
   - Model: {deploy_name}
3. The model will appear in the AI Assistant panel
   with full tool-calling capability.

USING WITH ATOMIC CHAT
-----------------------
1. Open Atomic Chat
2. Import {deploy_name}.gguf
3. The model will automatically be recognized as
   tool-capable (function-calling metadata is baked in).

USING WITH OLLAMA (import)
---------------------------
1. Copy {deploy_name}.gguf to your Ollama models directory
2. Create a Modelfile:
   FROM ./{deploy_name}.gguf
3. Run: ollama create custom-agent -f Modelfile
4. Run: ollama run custom-agent

API ENDPOINTS
--------------
POST /v1/chat/completions  — Chat with tool support
GET  /v1/models            — List available models
GET  /chat                 — Web chat interface
GET  /health               — Health check

MCP TOOLS AVAILABLE
--------------------
This model has {len(generate_tool_schemas())} built-in tool definitions
covering web fetching, Debian package management, database queries,
capability harvesting, and MCP server connections.

The tool schemas are baked into the GGUF metadata and are recognized
by any runtime that reads general.function_calling and the
{arch}.tool_calling_schema fields.

TROUBLESHOOTING
---------------
- Port {NEXUS_PORT} in use? Edit LAUNCH_SERVER.bat and change the port.
- Python not found? Install Python 3.10+ and ensure it's on PATH.
- Model not loading? Verify the GGUF file is not corrupted.
"""
    with open(os.path.join(DEPLOY_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(content)


def _generate_desktop_shortcut(deploy_name: str) -> dict[str, Any]:
    """Create a desktop shortcut to LAUNCH_SERVER.bat using PowerShell."""
    import subprocess

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_name = f"NEXUS Custom Agent ({deploy_name}).lnk"
    shortcut_path = os.path.join(desktop, shortcut_name)
    target = os.path.join(DEPLOY_DIR, "LAUNCH_SERVER.bat")
    icon = os.path.join(DEPLOY_DIR, f"{deploy_name}.gguf")  # fallback icon

    ps_script = f"""
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut('{shortcut_path}')
$s.TargetPath = '{target}'
$s.WorkingDirectory = '{DEPLOY_DIR}'
$s.Description = 'NEXUS Custom Agent Server (port {NEXUS_PORT})'
$s.Save()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, timeout=30, check=True,
        )
        return {"ok": True, "path": shortcut_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Server (standalone) ──────────────────────────────────────────────────────

def run_server(
    model_path: str,
    port: int = NEXUS_PORT,
    host: str = "127.0.0.1",
    n_ctx: int = 8192,
) -> None:
    """Start the OpenAI-compatible API server.

    This is a lightweight Flask server that:
    - Loads the forged GGUF with llama-cpp-python
    - Exposes /v1/chat/completions with tool_calls support
    - Serves a simple web chat UI at /chat
    - Provides health check at /health
    """
    try:
        from flask import Flask, jsonify, request, render_template_string
    except ImportError:
        print("Installing flask...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "-q"])
        from flask import Flask, jsonify, request, render_template_string

    from llama_cpp import Llama

    cfg_path = os.path.join(os.path.dirname(model_path), "..", "forge_cfg_real.json")
    hw = {}
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path, "r") as f:
            hw = json.load(f).get("hardware", {})

    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=hw.get("fit_min_context_size", n_ctx),
        n_batch=hw.get("ubatch_size", 512),
        n_threads=hw.get("threads", os.cpu_count()),
        n_threads_batch=hw.get("threads_batch", os.cpu_count()),
        rope_freq_base=hw.get("rope_frequency_base", 0.0),
        rope_freq_scale=hw.get("rope_frequency_scale_factor", 1.0),
        use_mmap=not hw.get("disable_mmap", False),
        use_mlock=hw.get("mlock", False),
        flash_attn=(hw.get("flash_attention", "Auto") == "Auto"),
        main_gpu=hw.get("main_gpu_index", 0),
        chat_format="chatml-function-calling",
        verbose=False,
    )

    app = Flask(__name__)

    CHAT_HTML = """
    <!DOCTYPE html>
    <html>
    <head><title>NEXUS Custom Agent</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #0a0a0a; color: #e0e0e0; }
        #chat { height: 60vh; overflow-y: auto; border: 1px solid #333; padding: 16px; border-radius: 12px; margin-bottom: 16px; background: #111; }
        .msg { margin: 8px 0; padding: 10px 14px; border-radius: 10px; }
        .user { background: #1a3a1a; border: 1px solid #2a5a2a; }
        .assistant { background: #1a1a3a; border: 1px solid #2a2a5a; }
        .tool { background: #2a2a1a; border: 1px solid #4a4a2a; font-family: monospace; font-size: 12px; }
        .role { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #39ff14; margin-bottom: 4px; }
        #input { width: calc(100% - 100px); padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; }
        #send { padding: 12px 24px; border-radius: 8px; border: none; background: #39ff14; color: #000; font-weight: bold; cursor: pointer; }
        #send:hover { background: #5aff3a; }
        .status { color: #888; font-size: 12px; margin: 4px 0; }
    </style>
    </head>
    <body>
        <h1 style="color:#39ff14;">NEXUS Custom Agent</h1>
        <div id="chat"></div>
        <div style="display:flex;gap:8px;">
            <input id="input" placeholder="Type a message..." onkeydown="if(event.key==='Enter') send()">
            <button id="send" onclick="send()">Send</button>
        </div>
        <script>
            let messageHistory = [];
            async function send() {
                const input = document.getElementById('input');
                const msg = input.value.trim();
                if (!msg) return;
                input.value = '';
                addMsg('user', msg);
                messageHistory.push({role: 'user', content: msg});
                await doAgentLoop();
            }
            
            async function doAgentLoop() {
                addMsg('status', 'Thinking...');
                try {
                    const res = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({model: 'custom-agent', messages: messageHistory, temperature: 0.4})
                    });
                    const data = await res.json();
                    document.querySelector('.status')?.remove();
                    
                    const message = data.choices?.[0]?.message;
                    if (!message) { addMsg('assistant', 'No response'); return; }
                    
                    messageHistory.push(message);
                    
                    if (message.content) {
                        addMsg('assistant', message.content);
                    }
                    
                    if (message.tool_calls && message.tool_calls.length > 0) {
                        for (const tc of message.tool_calls) {
                            addMsg('tool', 'Running tool: ' + tc.function.name + '(' + tc.function.arguments + ')');
                            const toolRes = await fetch('/v1/execute_tool', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({name: tc.function.name, arguments: tc.function.arguments, id: tc.id})
                            });
                            const toolData = await toolRes.json();
                            addMsg('tool', 'Result: ' + toolData.result);
                            messageHistory.push({
                                role: 'tool',
                                content: toolData.result,
                                name: tc.function.name,
                                tool_call_id: tc.id
                            });
                        }
                        await doAgentLoop();
                    }
                } catch(e) {
                    document.querySelector('.status')?.remove();
                    addMsg('assistant', 'Error: ' + e.message);
                }
            }
            
            function addMsg(role, text) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = 'msg ' + role;
                if (role === 'status') div.className = 'status';
                else div.innerHTML = '<div class="role">' + role + '</div>' + text.replace(/\\n/g, '<br>');
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model": os.path.basename(model_path), "port": port})

    @app.route("/v1/models")
    def list_models():
        return jsonify({
            "object": "list",
            "data": [{"id": "custom-agent", "object": "model", "created": 0, "owned_by": "nexus"}],
        })

    @app.route("/chat")
    def chat_ui():
        return render_template_string(CHAT_HTML)

    @app.route("/v1/execute_tool", methods=["POST"])
    def execute_tool():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        args_str = data.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except:
            args = {}
            
        result = f"Tool {name} not found."
        
        if name == "osint_catalog_search":
            query = args.get("query", "").lower()
            readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AWESOME_OSINT", "awesome-osint-master", "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                matches = []
                for line in lines:
                    if query in line.lower() and ("- [" in line or "http" in line):
                        matches.append(line.strip())
                if matches:
                    result = "\\n".join(matches[:15]) # Limit to top 15 matches to save context window
                else:
                    result = f"No OSINT tools found for '{query}'."
            else:
                result = "Awesome OSINT catalog not found on disk."
                
        elif name == "connect_mcp_server":
            result = f"SYSTEM NOTIFICATION: Live MCP hot-swapping requires Atomic Chat host environment. Native Flask fallback only supports pre-compiled OSINT tools. Do not simulate a connection."
        elif name == "disconnect_mcp_server":
            result = f"SYSTEM NOTIFICATION: Disconnect operation restricted in Native Flask wrapper. Switch to Atomic Chat for full server lifecycles."
            
        return jsonify({"result": result})

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.4)
        max_tokens = data.get("max_tokens", 1024)
        tools = data.get("tools", None)

        try:
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            return jsonify(resp)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"\n  Server running at http://{host}:{port}")
    print(f"  API:      http://{host}:{port}/v1/chat/completions")
    print(f"  Chat UI:  http://{host}:{port}/chat")
    print(f"  Health:   http://{host}:{port}/health")
    print(f"\n  Press Ctrl+C to stop.\n")
    app.run(host=host, port=port, debug=False)


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the forge toolchain."""
    import argparse

    parser = argparse.ArgumentParser(description="NEXUS GGUF Toolchain Forge")
    parser.add_argument("in_file", help="Source GGUF file path")
    parser.add_argument("--out", "-o", default=None, help="Output GGUF path (default: DEPLOY/custom-agent.gguf)")
    parser.add_argument("--name", default="custom-agent", help="Deployment name")
    parser.add_argument("--persona", default="", help="Persona core text")
    parser.add_argument("--context", type=int, default=None, help="Context window override")
    parser.add_argument("--precision", default="Q8_0", choices=["F32", "Q8_0"], help="Edited tensor precision")
    parser.add_argument("--no-tools", action="store_true", help="Skip tool schema injection")
    parser.add_argument("--deploy", action="store_true", help="Copy to DEPLOY/ and create shortcuts")
    parser.add_argument("--serve", action="store_true", help="Start the API server after forging")
    parser.add_argument("--port", type=int, default=NEXUS_PORT, help="Server port")
    parser.add_argument("--config", default=None, help="JSON config file path")

    args = parser.parse_args()

    # Load config if provided
    lever_values = {}
    surgery_intensities = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        rankings = cfg.get("rankings", {})
        defi_audit = float(rankings.get("defi_audit", rankings.get("finance", 0)))
        offensive_security = float(rankings.get("offensive_security", rankings.get("cyber", 0)))
        exploit_engineering = float(rankings.get("exploit_engineering", 0))
        stealth_ops = float(rankings.get("stealth_ops", 0))
        aggression = float(rankings.get("aggression", rankings.get("hostility", 0)))
        reasoning = float(rankings.get("reasoning", max(rankings.get("data", 0), rankings.get("math", 0))))
        creativity = float(rankings.get("creativity", rankings.get("creative", 0)))
        loyalty = float(rankings.get("loyalty", 100))
        lever_values = {
            "audit": defi_audit,
            "redteam": offensive_security,
            "exploit": exploit_engineering,
            "stealth": stealth_ops,
            "aggression": aggression,
            "depth": reasoning,
            "venom_loyalty": loyalty,
        }
        surgery_intensities = {
            "attn_focus": max(aggression, exploit_engineering * 0.3),
            "ffn_gain": max(reasoning, defi_audit, exploit_engineering),
            "output_chaos": creativity,
            "embd_perturb": max(creativity * 0.5, stealth_ops * 0.4),
            "output_aggression": aggression,
        }
        if args.persona == "" and cfg.get("directive"):
            args.persona = cfg["directive"]

    out_file = args.out or os.path.join(DEPLOY_DIR, f"{args.name}.gguf")

    def progress(frac, msg):
        bar = "█" * int(frac * 40) + "░" * (40 - int(frac * 40))
        print(f"\r[{bar}] {frac*100:5.1f}%  {msg}", end="", flush=True)
        if frac >= 1.0:
            print()

    print(f"\nNEXUS GGUF Toolchain Forge")
    print(f"  Source: {args.in_file}")
    print(f"  Output: {out_file}")
    print(f"  Tools:  {'yes' if not args.no_tools else 'no'}")
    print()

    result = forge_toolchain_gguf(
        in_file=args.in_file,
        out_file=out_file,
        persona_core=args.persona,
        lever_values=lever_values,
        surgery_intensities=surgery_intensities,
        context_window=args.context,
        edit_dtype=args.precision,
        include_tool_schemas=not args.no_tools,
        progress_cb=progress,
    )

    print(f"\n  Forge {'OK' if result['ok'] else 'FAILED'}")
    print(f"  Meta:     {'OK' if result['meta_ok'] else 'FAIL'}")
    print(f"  Surgery:  {'OK' if result['surgery_ok'] else 'FAIL'}")
    print(f"  Tool MD:  {'OK' if result['tool_ok'] else 'FAIL'}")
    print(f"  Size:     {os.path.getsize(out_file) / 1e9:.2f} GB")

    if args.deploy and result["ok"]:
        print("\n  Deploying...")
        dep = deploy_forged_model(out_file, args.name)
        print(f"  Deploy dir: {dep['deploy_dir']}")
        print(f"  Shortcut:   {'OK' if dep.get('shortcut_ok') else 'FAIL'}")
        print(f"\n  Run LAUNCH_SERVER.bat in {dep['deploy_dir']} to start.")

    if args.serve and result["ok"]:
        print(f"\n  Starting server on port {args.port}...")
        run_server(out_file, port=args.port)


if __name__ == "__main__":
    main()
