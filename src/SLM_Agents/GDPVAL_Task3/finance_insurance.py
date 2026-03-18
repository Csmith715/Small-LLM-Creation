from dataclasses import dataclass, field
from typing import Optional, List
import time
import pandas as pd
import requests
import json
from litellm import completion
from dotenv import load_dotenv
from src.SLM_Agents.agent_utils import clean_messages_for_model
import os

load_dotenv()
fmp_api_key = os.getenv('FMP_API_KEY')

@dataclass
class ECMState:
    as_of_date: str = "2025-04-11"  # required based on gdpval task details
    universe_df: Optional[pd.DataFrame] = None
    company_df: Optional[pd.DataFrame] = None
    subsector_df: Optional[pd.DataFrame] = None
    output_company_df: Optional[pd.DataFrame] = None
    output_subsector_df: Optional[pd.DataFrame] = None
    missing_tickers: List[str] = field(default_factory=list)


def safe_get_json(url: str, params: dict | None = None, timeout: int = 30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def weighted_avg(values: pd.Series, weights: pd.Series):
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return pd.NA
    v = values[mask]
    w = weights[mask]
    return (v * w).sum() / w.sum()

def load_sp500_universe(agent_state: ECMState) -> str:
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].copy()
    table.columns = [str(c).strip() for c in table.columns]

    rename_map = {
        "Symbol": "Ticker",
        "Security": "Company",
        "GICS Sector": "Sector",
        "GICS Sub-Industry": "SubSector",
    }
    table = table.rename(columns=rename_map)

    keep_cols = ["Ticker", "Company", "Sector", "SubSector"]
    table = table[keep_cols].copy()

    # Standardize tickers for APIs that prefer '-' over '.'
    table["TickerAPI"] = table["Ticker"].astype(str).str.replace(".", "-", regex=False)

    agent_state.universe_df = table

    return json.dumps({
        "ok": True,
        "num_companies": int(len(table)),
        "columns": list(table.columns),
        "sample_subsectors": sorted(table["SubSector"].dropna().unique().tolist())[:10]
    })

class ECMTools:
    def __init__(self, fi_state: ECMState, fmp_api: str):
        self.state = fi_state
        self.fmp_api_key = fmp_api

    def load_sp500_universe(self):
        return load_sp500_universe(self.state)

    def fetch_company_metrics_batch(self, batch_size: int = 50, start_idx: int = 0):
        if self.state.universe_df is None:
            return json.dumps({"ok": False, "error": "Universe not loaded."})

        df = self.state.universe_df.copy()
        tickers = df["TickerAPI"].tolist()
        batch = tickers[start_idx:start_idx + batch_size]

        if not batch:
            return json.dumps({"ok": True, "done": True, "rows_added": 0})

        joined = ",".join(batch)

        profile_url = f"https://financialmodelingprep.com/stable/profile-bulk"
        ratios_url = f"https://financialmodelingprep.com/stable/key-metrics-ttm-bulk"
        estimates_url = f"https://financialmodelingprep.com/stable/analyst-estimates-bulk"

        params = {"symbol": joined, "apikey": self.fmp_api_key}

        try:
            profile_data = safe_get_json(profile_url, params=params)
        except Exception as e:
            return json.dumps({"ok": False, "error": f"profile fetch failed: {e}"})

        # Ratios and estimates vary by provider. I may need to make these optional
        try:
            ratios_data = safe_get_json(ratios_url, params=params)
        except Exception as e:
            print(f"ratios fetch failed: {e}")
            ratios_data = []

        try:
            estimates_data = safe_get_json(estimates_url, params=params)
        except Exception as e:
            print(f"estimates fetch failed: {e}")
            estimates_data = []

        profile_map = {row.get("symbol"): row for row in profile_data if isinstance(row, dict)}
        ratios_map = {row.get("symbol"): row for row in ratios_data if isinstance(row, dict)}
        estimates_map = {row.get("symbol"): row for row in estimates_data if isinstance(row, dict)}

        rows = []
        missing = []

        for t in batch:
            p = profile_map.get(t)
            if not p:
                missing.append(t)
                continue

            r = ratios_map.get(t, {})
            e = estimates_map.get(t, {})

            rows.append({
                "TickerAPI": t,
                "MarketCap": p.get("mktCap"),
                "DividendYield": p.get("lastDiv") if p.get("price") else None,
                "Price": p.get("price"),
                "TrailingPE": p.get("pe") or r.get("peRatioTTM"),
                "ForwardPE": e.get("forwardPE"),
                "AnnualEPS_CY1": e.get("estimatedEpsAvg"),
                "QuarterlyEPS_CQ1": e.get("estimatedEpsNextQuarter"),
            })

        batch_df = pd.DataFrame(rows)
        if not batch_df.empty:
            if self.state.company_df is None:
                self.state.company_df = batch_df
            else:
                self.state.company_df = pd.concat(
                    [self.state.company_df, batch_df], ignore_index=True
                )

        self.state.missing_tickers.extend(missing)

        return json.dumps({
            "ok": True,
            "rows_added": int(len(batch_df)),
            "missing_tickers": missing[:10],
            "next_start_idx": start_idx + batch_size,
            "done": start_idx + batch_size >= len(tickers)
        })

    def compute_company_weights(self):
        if self.state.universe_df is None:
            return json.dumps({"ok": False, "error": "Universe not loaded."})
        if self.state.company_df is None or self.state.company_df.empty:
            return json.dumps({"ok": False, "error": "Company metrics not loaded."})

        df = self.state.universe_df.merge(
            self.state.company_df,
            on="TickerAPI",
            how="left"
        ).copy()

        numeric_cols = [
            "MarketCap", "DividendYield", "Price",
            "TrailingPE", "ForwardPE", "AnnualEPS_CY1", "QuarterlyEPS_CQ1"
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        total_market_cap = df["MarketCap"].sum(skipna=True)
        df["PctOfIndex"] = df["MarketCap"] / total_market_cap if total_market_cap else pd.NA
        df["NumCompanies"] = 1

        # Useful sorting flags
        df["TrailingPE_vs_15_20"] = pd.cut(
            df["TrailingPE"],
            bins=[-float("inf"), 15, 20, float("inf")],
            labels=["Below 15x", "15x-20x", "Above 20x"]
        )

        df["ForwardPE_vs_15_20"] = pd.cut(
            df["ForwardPE"],
            bins=[-float("inf"), 15, 20, float("inf")],
            labels=["Below 15x", "15x-20x", "Above 20x"]
        )

        self.state.output_company_df = df

        return json.dumps({
            "ok": True,
            "num_rows": int(len(df)),
            "total_market_cap": float(total_market_cap) if pd.notna(total_market_cap) else None
        })

    def compute_subsector_rollups(self):
        if self.state.output_company_df is None or self.state.output_company_df.empty:
            return json.dumps({"ok": False, "error": "Company output not ready."})

        df = self.state.output_company_df.copy()
        total_market_cap = df["MarketCap"].sum(skipna=True)

        rows = []
        for subsector, grp in df.groupby("SubSector", dropna=False):
            row = {
                "SubSector": subsector,
                "NoOfCompanies": int(len(grp)),
                "MarketCap": grp["MarketCap"].sum(skipna=True),
                "PctOfIndex": grp["MarketCap"].sum(skipna=True) / total_market_cap if total_market_cap else pd.NA,
                "TrailingPE": weighted_avg(grp["TrailingPE"], grp["MarketCap"]),
                "ForwardPE": weighted_avg(grp["ForwardPE"], grp["MarketCap"]),
                "DividendYield": weighted_avg(grp["DividendYield"], grp["MarketCap"]),
                "AnnualEPS_CY1": weighted_avg(grp["AnnualEPS_CY1"], grp["MarketCap"]),
                "QuarterlyEPS_CQ1": weighted_avg(grp["QuarterlyEPS_CQ1"], grp["MarketCap"]),
            }
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("MarketCap", ascending=False).reset_index(drop=True)

        for col in ["TrailingPE", "ForwardPE"]:
            out[f"{col}_vs_15_20"] = pd.cut(
                out[col],
                bins=[-float("inf"), 15, 20, float("inf")],
                labels=["Below 15x", "15x-20x", "Above 20x"]
            )

        self.state.subsector_df = out
        self.state.output_subsector_df = out

        return json.dumps({
            "ok": True,
            "num_subsectors": int(len(out))
        })

    def finalize_excel_tables(self):
        if self.state.output_company_df is None:
            return json.dumps({"ok": False, "error": "Company table not ready."})
        if self.state.output_subsector_df is None:
            return json.dumps({"ok": False, "error": "Subsector table not ready."})

        company_cols = [
            "Ticker", "Company", "Sector", "SubSector",
            "TrailingPE", "ForwardPE", "DividendYield",
            "AnnualEPS_CY1", "QuarterlyEPS_CQ1",
            "MarketCap", "NumCompanies", "PctOfIndex",
            "TrailingPE_vs_15_20", "ForwardPE_vs_15_20"
        ]

        subsector_cols = [
            "SubSector", "TrailingPE", "ForwardPE", "DividendYield",
            "AnnualEPS_CY1", "QuarterlyEPS_CQ1", "MarketCap",
            "NoOfCompanies", "PctOfIndex",
            "TrailingPE_vs_15_20", "ForwardPE_vs_15_20"
        ]

        # company_df = self.state.output_company_df[company_cols].copy()
        # subsector_df = self.state.output_subsector_df[subsector_cols].copy()

        self.state.output_company_df = self.state.output_company_df[company_cols].copy()
        self.state.output_subsector_df = self.state.output_subsector_df[subsector_cols].copy()

        return json.dumps({
            "ok": True,
            "company_rows": self.state.output_company_df.shape[0],
            "subsector_rows": self.state.output_subsector_df.shape[0],
            "missing_tickers_count": int(len(set(self.state.missing_tickers)))
        })


tools = [
    {
        "type": "function",
        "function": {
            "name": "load_sp500_universe",
            "description": "Load the S&P 500 universe with ticker, company, sector, and sub-sector.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_company_metrics_batch",
            "description": "Fetch market data and estimate fields for a batch of S&P 500 companies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer"},
                    "start_idx": {"type": "integer"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_company_weights",
            "description": "Merge universe and metrics, then compute company-level percent of index.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_subsector_rollups",
            "description": "Aggregate company metrics into sub-sector level weighted summaries.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_excel_tables",
            "description": "Prepare the final company and sub-sector output tables.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]

ecm_state = ECMState(as_of_date="2025-04-11")
tool_obj = ECMTools(fi_state=ecm_state, fmp_api=fmp_api_key)

tool_registry = {
    "load_sp500_universe": tool_obj.load_sp500_universe,
    "fetch_company_metrics_batch": tool_obj.fetch_company_metrics_batch,
    "compute_company_weights": tool_obj.compute_company_weights,
    "compute_subsector_rollups": tool_obj.compute_subsector_rollups,
    "finalize_excel_tables": tool_obj.finalize_excel_tables,
}

def run_ecm_pipeline(tool_reg, state, batch_size=50):
    print(tool_reg["load_sp500_universe"]())

    n = len(state.universe_df)
    for start_idx in range(0, n, batch_size):
        print(tool_reg["fetch_company_metrics_batch"](batch_size=batch_size, start_idx=start_idx))
        time.sleep(0.25)

    print(tool_reg["compute_company_weights"]())
    print(tool_reg["compute_subsector_rollups"]())
    print(tool_reg["finalize_excel_tables"]())

    return {
        "company_df": state.output_company_df,
        "subsector_df": state.output_subsector_df,
        "missing_tickers": sorted(set(state.missing_tickers)),
    }

def build_ecm_progress_summary(flags, state):
    if not flags["universe_loaded"]:
        next_step = "Call load_sp500_universe."
    elif not flags["metrics_fetched"]:
        next_step = "Fetch company metrics in batches."
    elif not flags["company_weights_computed"]:
        next_step = "Call compute_company_weights."
    elif not flags["subsector_rollups_computed"]:
        next_step = "Call compute_subsector_rollups."
    elif not flags["tables_finalized"]:
        next_step = "Call finalize_excel_tables."
    else:
        next_step = "Finish with a brief summary."

    company_rows = 0 if state.company_df is None else len(state.company_df)
    universe_rows = 0 if state.universe_df is None else len(state.universe_df)
    subsector_rows = 0 if state.output_subsector_df is None else len(state.output_subsector_df)

    return {
        "role": "system",
        "content": (
            "Execution status:\n"
            f"- Universe loaded: {flags['universe_loaded']}\n"
            f"- Company metrics fetched: {flags['metrics_fetched']}\n"
            f"- Company rows collected: {company_rows}\n"
            f"- Universe size: {universe_rows}\n"
            f"- Company weights computed: {flags['company_weights_computed']}\n"
            f"- Sub-sector rollups computed: {flags['subsector_rollups_computed']}\n"
            f"- Final tables ready: {flags['tables_finalized']}\n"
            f"- Current sub-sector rows: {subsector_rows}\n"
            f"- Next step: {next_step}\n\n"
            "Use only exact tool names from the provided tool list.\n"
            "Do not invent tool names.\n"
            "Do not use status labels as tool names.\n"
            "Do not use tool call IDs as tool names.\n"
            "Keep tool use minimal and move to the next unfinished step."
        )
    }


def allowed_ecm_tools_for_stage(flags):
    if not flags["universe_loaded"]:
        return {"load_sp500_universe"}
    if not flags["metrics_fetched"]:
        return set()
    if not flags["company_weights_computed"]:
        return {"compute_company_weights"}
    if not flags["subsector_rollups_computed"]:
        return {"compute_subsector_rollups"}
    if not flags["tables_finalized"]:
        return {"finalize_excel_tables"}
    return set()


def run_ecm_agentic_task(
    messages,
    agent_tools,
    agent_tool_registry,
    state,
    model="ollama/qwen2.5:1.5b-instruct",
    batch_size=50,
    max_steps=20,
):
    flags = {
        "universe_loaded": False,
        "metrics_fetched": False,
        "company_weights_computed": False,
        "subsector_rollups_computed": False,
        "tables_finalized": False,
    }

    for step in range(max_steps):
        print(f"\nSTEP {step + 1}")

        # deterministic batch-fetch step for small models
        if flags["universe_loaded"] and not flags["metrics_fetched"]:
            n = len(state.universe_df) if state.universe_df is not None else 0
            for start_idx in range(0, n, batch_size):
                print(f"--- Agent calling fetch_company_metrics_batch with: "
                      f"{{'batch_size': {batch_size}, 'start_idx': {start_idx}}} ---")
                try:
                    obs = agent_tool_registry["fetch_company_metrics_batch"](
                        batch_size=batch_size,
                        start_idx=start_idx
                    )
                except Exception as e:
                    return {
                        "message": f"Batch fetch failed: {e}",
                        "company_df": getattr(state, "output_company_df", None),
                        "subsector_df": getattr(state, "output_subsector_df", None),
                        "missing_tickers": getattr(state, "missing_tickers", []),
                    }

                messages.append({
                    "role": "tool",
                    "name": "fetch_company_metrics_batch",
                    "content": obs
                })
                print(f"--- Tool result: {obs[:250]} ---")

            flags["metrics_fetched"] = True
            continue

        progress_msg = build_ecm_progress_summary(flags, state)
        allowed = allowed_ecm_tools_for_stage(flags)

        filtered_tools = [
            t for t in agent_tools
            if t["function"]["name"] in allowed and t["function"]["name"] in agent_tool_registry
        ]

        request_messages = clean_messages_for_model(
            [messages[0], progress_msg] + messages[1:]
        )

        response = completion(
            model=model,
            messages=request_messages,
            tools=filtered_tools if filtered_tools else None,
            tool_choice="auto" if filtered_tools else "none",
            temperature=0,
        )

        message = response.choices[0].message

        assistant_msg = {
            "role": "assistant",
            "content": message.content or ""
        }

        if getattr(message, "tool_calls", None):
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}"
                    }
                }
                for tc in message.tool_calls
            ]

        messages.append(assistant_msg)

        if not getattr(message, "tool_calls", None):
            return {
                "message": message.content or "Workflow complete.",
                "company_df": getattr(state, "output_company_df", None),
                "subsector_df": getattr(state, "output_subsector_df", None),
                "missing_tickers": sorted(set(getattr(state, "missing_tickers", []))),
            }

        for tool_call in message.tool_calls:
            raw_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                observation = json.dumps({
                    "ok": False,
                    "error": "Invalid JSON arguments."
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": raw_name,
                    "content": observation
                })
                continue

            print(f"--- Agent calling {raw_name} with: {args} ---")

            if raw_name not in agent_tool_registry:
                observation = json.dumps({
                    "ok": False,
                    "error": "Invalid tool name. Use only allowed tools."
                })
            elif raw_name not in allowed:
                observation = json.dumps({
                    "ok": False,
                    "error": "Tool not allowed right now. Use the next unfinished step."
                })
            else:
                try:
                    result = agent_tool_registry[raw_name](**args)
                    observation = result if isinstance(result, str) else json.dumps(result)
                except Exception as e:
                    observation = json.dumps({
                        "ok": False,
                        "error": f"Tool failed: {str(e)}"
                    })

            print(f"--- Tool result: {observation[:250]} ---")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": raw_name,
                "content": observation
            })

            try:
                obs = json.loads(observation) if isinstance(observation, str) else observation
            except Exception as e:
                print(f"--- Tool result failed to parse: {e} ---")
                obs = {}

            if raw_name == "load_sp500_universe" and obs.get("ok") is True:
                flags["universe_loaded"] = True

            elif raw_name == "compute_company_weights" and obs.get("ok") is True:
                flags["company_weights_computed"] = True

            elif raw_name == "compute_subsector_rollups" and obs.get("ok") is True:
                flags["subsector_rollups_computed"] = True

            elif raw_name == "finalize_excel_tables" and obs.get("ok") is True:
                flags["tables_finalized"] = True

        if all(flags.values()):
            return {
                "message": "Workflow complete.",
                "company_df": getattr(state, "output_company_df", None),
                "subsector_df": getattr(state, "output_subsector_df", None),
                "missing_tickers": sorted(set(getattr(state, "missing_tickers", []))),
            }

    return {
        "message": f"Stopped after {max_steps} steps.",
        "company_df": getattr(state, "output_company_df", None),
        "subsector_df": getattr(state, "output_subsector_df", None),
        "missing_tickers": sorted(set(getattr(state, "missing_tickers", []))),
    }
