from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from litellm import completion
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from src.SLM_Agents.agent_utils import clean_messages_for_model


@dataclass
class BrandDocState:
    title: str = "Brand Data Gathering"
    draft_text: Optional[str] = None
    pdf_path: Optional[str] = None
    target_pages: int = 3
    min_words: int = 900
    max_words: int = 1800


class BrandDocTools:
    def __init__(self, state: BrandDocState):
        self.state = state

    # I may want to change this
    def draft_brand_data_gathering_content(self) -> str:
        """
        This Creates the document content deterministically.
        This is intentionally not delegated to the model because the task is stable, and
        we want a predictable output for a small SLM pipeline.
        """
        text = f"""{self.state.title}

Purpose: This document is used to gather operational, sales, logistics, and commercial information from potential or newly onboarded brand partners. The information collected 
will help internal teams evaluate brand readiness for distribution, assess operational capacity, understand fulfillment requirements, and prepare for successful integration.

Please answer each question as completely as possible. If a question does not apply, write N/A. If additional detail is needed, please continue your answer beneath the question.

Brand Overview

1. What is the full legal name of the brand or company?
________________________________________________________________________________

2. What is the brand's primary headquarters address?
________________________________________________________________________________

3. Who are the primary contacts for this onboarding process, including name, title, email address, and phone number?
________________________________________________________________________________
________________________________________________________________________________

4. Please provide a short overview of the brand, including product focus, target customer, and current go-to-market strategy.
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

5. What channels does the brand currently sell through, including direct-to-consumer, retail, wholesale, marketplace, or other channels?
________________________________________________________________________________
________________________________________________________________________________

6. What geographic regions does the brand currently serve?
________________________________________________________________________________

7. Has the brand worked with distributors before? If yes, please describe the distributor relationships and markets covered.
________________________________________________________________________________
________________________________________________________________________________

Sales and Commercial Readiness

8. What are the brand's top-selling products or core SKUs?
________________________________________________________________________________
________________________________________________________________________________

9. Please provide a current product catalog or summarize the active assortment by category.
________________________________________________________________________________
________________________________________________________________________________

10. What is the average wholesale selling price by key product line or SKU group?
________________________________________________________________________________

11. What is the suggested retail price by key product line or SKU group?
________________________________________________________________________________

12. Are there minimum order quantities by SKU, case, or order? If yes, please describe.
________________________________________________________________________________
________________________________________________________________________________

13. Are there channel restrictions, territorial restrictions, or customer exclusivity considerations?
________________________________________________________________________________
________________________________________________________________________________

14. What is the current annual revenue of the brand and what sales growth is expected over the next 12 months?
________________________________________________________________________________
________________________________________________________________________________

15. What are the brand's sales goals for the next 12 months by channel or region?
________________________________________________________________________________
________________________________________________________________________________

16. Are there specific priority customers, accounts, retailers, or regions the brand wants to target first?
________________________________________________________________________________
________________________________________________________________________________

17. What promotional programs, trade spend expectations, rebates, or discount structures are currently used?
________________________________________________________________________________
________________________________________________________________________________

18. Are there seasonal sales patterns, launch calendars, or major promotional periods that internal teams should know about?
________________________________________________________________________________
________________________________________________________________________________

Operational and Supply Chain Readiness

19. Where are products manufactured?
________________________________________________________________________________

20. What are the primary ship-from locations or warehouse locations?
________________________________________________________________________________

21. Does the brand use internal warehousing, a third-party logistics provider, or a hybrid model?
________________________________________________________________________________

22. What is the standard production lead time for replenishment orders?
________________________________________________________________________________

23. What is the current average order fill rate?
________________________________________________________________________________

24. What is the brand's current monthly or quarterly production capacity?
________________________________________________________________________________

25. Are there known production bottlenecks, raw material constraints, supplier risks, or other operational limitations?
________________________________________________________________________________
________________________________________________________________________________

26. What is the standard case pack, inner pack, pallet configuration, and shipping unit information for products?
________________________________________________________________________________
________________________________________________________________________________

27. Are product dimensions, weights, and master case details available for all active SKUs?
________________________________________________________________________________

28. Are barcodes, UPCs, GTINs, lot codes, expiration dates, or traceability requirements in place?
________________________________________________________________________________
________________________________________________________________________________

29. Are products temperature sensitive, fragile, hazardous, regulated, or subject to special storage or transportation requirements?
________________________________________________________________________________
________________________________________________________________________________

30. What are the standard order processing times from receipt of order to shipment?
________________________________________________________________________________

31. What are the brand's policies for backorders, substitutions, and discontinued items?
________________________________________________________________________________
________________________________________________________________________________

32. What are the return, damage, and claims handling policies?
________________________________________________________________________________
________________________________________________________________________________

Systems, Data, and Integration Readiness

33. What ERP, inventory management, warehouse management, EDI, or order management systems are currently used?
________________________________________________________________________________
________________________________________________________________________________

34. Can the brand support electronic data exchange, including EDI, CSV, API, portal-based order management, or other methods?
________________________________________________________________________________
________________________________________________________________________________

35. Can the brand provide regular inventory availability feeds, pricing files, and product master data updates?
________________________________________________________________________________
________________________________________________________________________________

36. Who is responsible for system integration, item setup, and ongoing data maintenance?
________________________________________________________________________________

37. Are product images, marketing assets, compliance documents, certifications, and sell sheets available?
________________________________________________________________________________
________________________________________________________________________________

38. Are there any data gaps, system limitations, or process constraints that could impact onboarding?
________________________________________________________________________________
________________________________________________________________________________

Customer Support and Account Management

39. Who will manage the distributor relationship on the brand side after onboarding?
________________________________________________________________________________

40. What is the escalation path for service issues, order issues, inventory shortages, pricing disputes, or operational delays?
________________________________________________________________________________
________________________________________________________________________________

41. What are the expected service levels for response times, issue resolution, and account support?
________________________________________________________________________________
________________________________________________________________________________

42. Are there training materials, brand guidelines, or product education resources available for internal teams?
________________________________________________________________________________
________________________________________________________________________________

Compliance and Risk Review

43. Are there insurance certificates, regulatory documents, testing reports, or compliance certifications required for distribution?
________________________________________________________________________________
________________________________________________________________________________

44. Are there legal, labeling, import, export, or market-specific compliance requirements internal teams should know about?
________________________________________________________________________________
________________________________________________________________________________

45. Are there any current litigation matters, recalls, quality concerns, or operational risks that may affect onboarding?
________________________________________________________________________________
________________________________________________________________________________

Final Readiness Assessment

46. What does the brand believe will be required for a successful launch with distribution?
________________________________________________________________________________
________________________________________________________________________________

47. What internal concerns or risks does the brand want to highlight in advance?
________________________________________________________________________________
________________________________________________________________________________

48. What timeline is the brand targeting for onboarding, item setup, first order, and first shipment?
________________________________________________________________________________
________________________________________________________________________________

49. What support does the brand expect from the distribution partner during onboarding and launch?
________________________________________________________________________________
________________________________________________________________________________

50. Is there any additional information that would help internal teams assess readiness, prepare operations, or support sales execution?
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
"""
        self.state.draft_text = text
        return json.dumps({
            "ok": True,
            "title": self.state.title,
            "word_count": len(text.split())
        })

    def review_document_length(self) -> str:
        if not self.state.draft_text:
            return json.dumps({"ok": False, "error": "No draft text exists."})

        word_count = len(self.state.draft_text.split())
        meets_length = self.state.min_words <= word_count <= self.state.max_words

        return json.dumps({
            "ok": True,
            "word_count": word_count,
            "min_words": self.state.min_words,
            "max_words": self.state.max_words,
            "length_ok": meets_length
        })

    def finalize_pdf(self, output_path: str = "Brand_Data_Gathering.pdf") -> str:
        if not self.state.draft_text:
            return json.dumps({"ok": False, "error": "No draft text exists."})

        path = Path(output_path).resolve()
        self._write_text_pdf(self.state.draft_text, str(path))
        self.state.pdf_path = str(path)

        return json.dumps({
            "ok": True,
            "pdf_path": self.state.pdf_path
        })

    def _write_text_pdf(self, text: str, output_path: str):
        c = canvas.Canvas(output_path, pagesize=LETTER)
        width, height = LETTER

        font_name = "Times-Roman"
        font_size = 11
        line_height = 14

        left_margin = 54
        right_margin = 54
        top_margin = 54
        bottom_margin = 54

        usable_width = width - left_margin - right_margin

        def wrap_paragraph(paragraph: str) -> list[str]:
            if not paragraph.strip():
                return [""]
            words = paragraph.split()
            para_lines = []
            current = ""

            for word in words:
                trial = word if not current else f"{current} {word}"
                if stringWidth(trial, font_name, font_size) <= usable_width:
                    current = trial
                else:
                    para_lines.append(current)
                    current = word
            if current:
                para_lines.append(current)
            return para_lines

        paragraphs = text.split("\n")
        lines = []
        for p in paragraphs:
            if p.strip().startswith("___"):
                lines.append(p)
            elif not p.strip():
                lines.append("")
            else:
                lines.extend(wrap_paragraph(p))

        y = height - top_margin
        c.setFont(font_name, font_size)

        page_no = 1
        for line in lines:
            if y < bottom_margin:
                c.setFont(font_name, 10)
                c.drawRightString(width - right_margin, 30, f"Page {page_no}")
                c.showPage()
                page_no += 1
                c.setFont(font_name, font_size)
                y = height - top_margin

            c.drawString(left_margin, y, line)
            y -= line_height

        c.setFont(font_name, 10)
        c.drawRightString(width - right_margin, 30, f"Page {page_no}")
        c.save()


wholesale_agent_tools = [
        {
            "type": "function",
            "function": {
                "name": "draft_brand_data_gathering_content",
                "description": "Create the full text content for the Brand Data Gathering PDF.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "review_document_length",
                "description": "Check whether the drafted document is an appropriate length.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_pdf",
                "description": "Generate the final text-based PDF document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string"}
                    },
                    "required": []
                }
            }
        }
    ]

def build_progress_summary(flags, state: BrandDocState):
    if not flags["draft_created"]:
        next_step = "Call draft_brand_data_gathering_content."
    elif not flags["length_reviewed"]:
        next_step = "Call review_document_length."
    elif not flags["pdf_finalized"]:
        next_step = "Call finalize_pdf."
    else:
        next_step = "Finish."

    word_count = len(state.draft_text.split()) if state.draft_text else 0

    return {
        "role": "system",
        "content": (
            "Execution status:\n"
            f"- Draft created: {flags['draft_created']}\n"
            f"- Length reviewed: {flags['length_reviewed']}\n"
            f"- PDF finalized: {flags['pdf_finalized']}\n"
            f"- Current word count: {word_count}\n"
            f"- Target title: {state.title}\n"
            f"- Next step: {next_step}\n\n"
            "Use only exact tool names from the tool list.\n"
            "Do not invent tool names.\n"
            "Do not use status labels as tool names.\n"
            "Do not use tool call ids as tool names.\n"
            "Move to the next unfinished step."
        )
    }


def allowed_tools_for_stage(flags):
    if not flags["draft_created"]:
        return {"draft_brand_data_gathering_content"}
    if not flags["length_reviewed"]:
        return {"review_document_length"}
    if not flags["pdf_finalized"]:
        return {"finalize_pdf"}
    return set()


def run_brand_doc_agentic_task(
    messages,
    tools,
    tool_registry,
    state,
    model="ollama/qwen2.5:1.5b-instruct",
    max_steps=10,
):
    flags = {
        "draft_created": False,
        "length_reviewed": False,
        "pdf_finalized": False,
    }

    for step in range(max_steps):
        print(f"\nSTEP {step + 1}")

        progress_msg = build_progress_summary(flags, state)
        allowed = allowed_tools_for_stage(flags)

        filtered_tools = [
            t for t in tools
            if t["function"]["name"] in allowed and t["function"]["name"] in tool_registry
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
                "draft_text": state.draft_text,
                "pdf_path": state.pdf_path,
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

            if raw_name not in tool_registry:
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
                    result = tool_registry[raw_name](**args)
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
                print(f"Error parsing tool result: {e}")
                obs = {}

            if raw_name == "draft_brand_data_gathering_content" and obs.get("ok") is True:
                flags["draft_created"] = True

            elif raw_name == "review_document_length" and obs.get("ok") is True:
                flags["length_reviewed"] = True
                if obs.get("length_ok") is True and not flags["pdf_finalized"]:
                    print("--- Auto-calling finalize_pdf with: {} ---")
                    final_obs = tool_registry["finalize_pdf"]()
                    print(f"--- Tool result: {final_obs[:250]} ---")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "finalize_pdf",
                        "content": final_obs
                    })
                    try:
                        final_parsed = json.loads(final_obs)
                    except Exception as e:
                        print(f"Error parsing tool result: {e}")
                        final_parsed = {}
                    if final_parsed.get("ok"):
                        flags["pdf_finalized"] = True

            elif raw_name == "finalize_pdf" and obs.get("ok") is True:
                flags["pdf_finalized"] = True

        if all(flags.values()):
            return {
                "message": "Workflow complete.",
                "draft_text": state.draft_text,
                "pdf_path": state.pdf_path,
            }

    return {
        "message": f"Stopped after {max_steps} steps.",
        "draft_text": state.draft_text,
        "pdf_path": state.pdf_path,
    }
