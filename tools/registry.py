from tools.search import search_web

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "인터넷에서 최신 정보를 검색합니다. "
                "날씨, 뉴스, 실시간 정보, 모르는 내용이 필요할 때 사용합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 또는 질문",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

_TOOL_FUNCTIONS = {
    "search_web": search_web,
}


def get_tool_prompt() -> str:
    lines = ["사용 가능한 도구:"]
    for td in TOOL_DEFINITIONS:
        f = td["function"]
        lines.append(f"- {f['name']}: {f['description']}")
    lines.append(
        "\n도구 사용 시 반드시 아래 형식만 사용하세요:\n"
        "<tool_call>\n"
        '{"name": "도구이름", "arguments": {"파라미터": "값"}}\n'
        "</tool_call>"
    )
    return "\n".join(lines)


def execute_tool(name: str, args: dict) -> str:
    func = _TOOL_FUNCTIONS.get(name)
    if not func:
        return f"알 수 없는 도구: {name}"
    return func(**args)
