def clean_messages_for_model(messages):
    cleaned = []
    for m in messages:
        role = m.get("role")
        if role in {"system", "user"}:
            cleaned.append(m)
        elif role == "assistant" and not m.get("tool_calls"):
            cleaned.append({"role": "assistant", "content": m.get("content", "")})
        elif role == "tool":
            content = m.get("content", "")
            if "Invalid tool name" in content:
                continue
            cleaned.append({
                "role": "tool",
                "name": m.get("name", ""),
                "content": content
            })
    return cleaned