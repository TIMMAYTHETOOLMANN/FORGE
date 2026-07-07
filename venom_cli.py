#!/usr/bin/env python
import sys
import os
import json
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gguf_engine as engine

def cmd_browse():
    try:
        import tkinter as tk
        import tkinter.filedialog as fd
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = fd.askopenfilename(
            title="Select Target GGUF File",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")]
        )
        # print to stdout
        if path:
            print(json.dumps({"success": True, "path": os.path.abspath(path)}))
        else:
            print(json.dumps({"success": True, "path": ""}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def cmd_scan(path):
    if not path or not os.path.exists(path):
        print(json.dumps({"success": False, "error": f"File not found: {path}"}))
        return
    try:
        reader = engine.read_gguf(path)
        arch = engine.get_architecture(reader)
        prec = engine.detect_precision(reader)
        
        # Get behavior catalog keys
        catalog = engine.resolve_behavior_catalog(arch)
        keys = [item["key"] for item in catalog]
        
        current_values = engine.read_current_values(reader, keys)
        
        # Safe close
        engine.close_reader(reader)
        
        print(json.dumps({
            "success": True,
            "path": os.path.abspath(path),
            "arch": arch,
            "precision": prec,
            "current_values": current_values
        }, default=str))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))

def cmd_forge(in_path, out_path, config_path):
    if not in_path or not os.path.exists(in_path):
        print(json.dumps({"success": False, "error": f"Input file not found: {in_path}"}))
        return
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        rankings = config.get("rankings", {})
        custom_directive = config.get("directive", "")
        hardware = config.get("hardware", {})
        
        # Map rankings to persona levers (0-100)
        # Eight non-redundant ranking keys (SL8)
        defi_audit = float(rankings.get("defi_audit", rankings.get("finance", 0)))
        offensive_security = float(rankings.get("offensive_security", rankings.get("cyber", 0)))
        exploit_engineering = float(rankings.get("exploit_engineering", 0))
        stealth_ops = float(rankings.get("stealth_ops", 0))
        aggression = float(rankings.get("aggression", rankings.get("hostility", 0)))
        reasoning = float(rankings.get("reasoning", max(rankings.get("data", 0), rankings.get("math", 0))))
        creativity = float(rankings.get("creativity", rankings.get("creative", 0)))
        loyalty = float(rankings.get("loyalty", 100))  # default 100% Commander loyalty
        
        # Map to GGUF engine persona levers
        # VENOM_CORE is always composed at 100% by compose_directive() — non-optional
        lever_values = {
            "audit": defi_audit,
            "redteam": offensive_security,
            "exploit": exploit_engineering,
            "stealth": stealth_ops,
            "aggression": aggression,
            "depth": reasoning,
            "venom_loyalty": loyalty,
        }
        
        # Map to weight surgery intensities
        surgery_intensities = {
            "attn_focus": max(aggression, exploit_engineering * 0.3),
            "ffn_gain": max(reasoning, defi_audit, exploit_engineering),
            "output_chaos": creativity,
            "embd_perturb": max(creativity * 0.5, stealth_ops * 0.4),
            "output_aggression": aggression,
        }
        
        # Build the composed system directive
        directive = engine.compose_directive(custom_directive, lever_values)
        chat_template = engine.compose_chat_template(directive)
        
        # Prepare modified config (metadata updates)
        modified_config = {}
        injected_fields = {}
        
        # Inject chat template
        injected_fields["tokenizer.chat_template"] = {
            "value": chat_template,
            "type": engine.GGUFValueType.STRING
        }
        
        # If hardware specifies context window, update it
        reader = engine.read_gguf(in_path)
        arch = engine.get_architecture(reader)
        
        ctx_key = f"{arch}.context_length"
        if "context_window" in hardware:
            try:
                ctx_val = int(hardware["context_window"])
                modified_config[ctx_key] = ctx_val
            except Exception:
                pass
                
        # Resolve active weight surgeries
        tensor_ops = []
                
        def progress_cb(frac, msg):
            # Print structured progress to stdout so Node.js can parse in real-time
            print(f"PROGRESS: {frac:.4f} : {msg}", flush=True)
            
        # Storage precision for edited tensors: "Q8_0" keeps output near source
        # size (compact), "F32" is lossless but ~4-5x larger. Default Q8_0.
        edit_dtype = str(hardware.get("edit_precision", "Q8_0")).upper()
        if edit_dtype not in ("F32", "Q8_0"):
            edit_dtype = "Q8_0"

        if tensor_ops:
            # Requires full weight surgery
            report = engine.write_surgical_gguf(
                in_path,
                out_path,
                reader,
                tensor_ops,
                modified_config=modified_config,
                injected_fields=injected_fields,
                progress_cb=progress_cb,
                edit_dtype=edit_dtype
            )
            # Verify weight surgery
            verify_report = engine.verify_weight_surgery(out_path, report.get("edited", {}))
            surgery_ok = verify_report.get("ok", True)
        else:
            # Metadata-only write
            engine.write_modified_gguf(
                in_path,
                out_path,
                reader,
                modified_config=modified_config,
                injected_fields=injected_fields,
                progress_cb=progress_cb
            )
            surgery_ok = True
            
        # Verify metadata changes
        meta_report = engine.verify_changes(out_path, modified_config, injected_fields)
        meta_ok = meta_report.get("ok", True)
        
        engine.close_reader(reader)
        
        print(json.dumps({
            "success": surgery_ok and meta_ok,
            "surgery_ok": surgery_ok,
            "meta_ok": meta_ok,
            "out_file": os.path.abspath(out_path),
            "meta_report": meta_report.get("checks", [])
        }))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))

def cmd_forge_toolchain(in_path, config_path):
    """Forge a tool-calling-capable GGUF using forge_toolchain.py."""
    if not in_path or not os.path.exists(in_path):
        print(json.dumps({"success": False, "error": f"Input file not found: {in_path}"}))
        return
    try:
        import forge_toolchain as ft

        # Load config
        lever_values = {}
        surgery_intensities = {}
        persona_core = ""
        context_window = None
        edit_dtype = "Q8_0"
        include_tools = True
        deploy_name = "custom-agent"
        auto_deploy = False
        auto_serve = False
        port = ft.NEXUS_PORT

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            rankings = cfg.get("rankings", {})
            # Eight non-redundant ranking keys (SL8)
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
            persona_core = cfg.get("directive", "")
            tc = cfg.get("toolchain", {})
            include_tools = tc.get("inject_tool_schemas", True)
            deploy_name = tc.get("deploy_name", "custom-agent")
            edit_dtype = tc.get("edit_precision", "Q8_0")
            ctx = tc.get("context_window", 0)
            if ctx:
                context_window = int(ctx)
            auto_deploy = tc.get("auto_deploy", False)
            auto_serve = tc.get("auto_serve", False)
            port = int(tc.get("port", ft.NEXUS_PORT))

        out_file = os.path.join(ft.DEPLOY_DIR, f"{deploy_name}.gguf")

        def progress_cb(frac, msg):
            print(f"PROGRESS: {frac:.4f} : {msg}", flush=True)

        result = ft.forge_toolchain_gguf(
            in_file=in_path,
            out_file=out_file,
            persona_core=persona_core,
            lever_values=lever_values,
            surgery_intensities=surgery_intensities,
            context_window=context_window,
            edit_dtype=edit_dtype,
            include_tool_schemas=include_tools,
            progress_cb=progress_cb,
        )

        if result.get("ok") and auto_deploy:
            dep = ft.deploy_forged_model(out_file, deploy_name)
            result["deploy"] = dep

        if result.get("ok") and auto_serve:
            print(f"SERVER_START: {port}")
            ft.run_server(out_file, port=port)

        print(json.dumps({
            "success": result.get("ok", False),
            "meta_ok": result.get("meta_ok"),
            "surgery_ok": result.get("surgery_ok"),
            "tool_ok": result.get("tool_ok"),
            "out_file": result.get("out_file"),
            "arch": result.get("arch"),
            "tensor_ops": result.get("tensor_ops"),
            "deploy": result.get("deploy"),
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))


def cmd_setup_python():
    """Configure Python interpreter for IntelliJ IDEA."""
    try:
        import setup_python
        result = setup_python.configure_idea()
        print(json.dumps(result))
    except ImportError:
        # setup_python not yet created; provide manual instructions
        venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
        print(json.dumps({
            "success": os.path.exists(venv),
            "venv_python": venv if os.path.exists(venv) else None,
            "hint": "Run: python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt",
        }))


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No command provided"}))
        return
        
    cmd = sys.argv[1]
    if cmd == "browse":
        cmd_browse()
    elif cmd == "scan":
        if len(sys.argv) < 3:
            print(json.dumps({"success": False, "error": "No file path provided for scan"}))
            return
        cmd_scan(sys.argv[2])
    elif cmd == "forge":
        if len(sys.argv) < 5:
            print(json.dumps({"success": False, "error": "Usage: forge <in_path> <out_path> <config_json_path>"}))
            return
        cmd_forge(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "forge_toolchain":
        if len(sys.argv) < 3:
            print(json.dumps({"success": False, "error": "Usage: forge_toolchain <in_path> [config_path]"}))
            return
        config_path = sys.argv[3] if len(sys.argv) > 3 else "forge_cfg_real.json"
        cmd_forge_toolchain(sys.argv[2], config_path)
    elif cmd == "setup_python":
        cmd_setup_python()
    else:
        print(json.dumps({"success": False, "error": f"Unknown command: {cmd}"}))

if __name__ == "__main__":
    main()
