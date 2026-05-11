import re
import logging
from typing import Generator
from llama_cpp import Llama
from tools.registry import execute_tool
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

_SEARCH_KEYWORDS = [
    "날씨", "기온", "강수", "미세먼지",
    "뉴스", "속보", "주식", "환율",
    "경기 결과", "순위", "영화", "드라마",
    "유튜브", "노래", "음악", "노래 추천"
]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _needs_search(text: str) -> bool:
    return any(kw in text for kw in _SEARCH_KEYWORDS)


def _strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text).strip()


class LLM:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict] = []
        self.memory = MemoryStore()
        self.system_prompt = (
            "당신은 친절한 한국어 인공지능 어시스턴트입니다. "
            "반드시 한국어로만 대답하세요. "
            "중국어(한자), 영어, 일본어, 특수기호(%, # 등)는 한글로 읽을 수 있게 표기하세요. "
            "숫자를 읽을 때는 반드시 한글로 표기하세요. (예: '23도', '100달러', '2024년' X → '이십삼도', '백달러', '이천이십사년' O)"
            # "간결하게 두세 문장으로 답변하세요."
        )

    def generate(self, user_input: str) -> Generator[str, None, None]:
        messages = [{"role": "system", "content": self.system_prompt}]

        past = ""
        if len(user_input) >= 5:
            past = self.memory.search(user_input)

        if past:
            messages.append({
                "role": "system",
                "content": f"[과거 대화 참고]\n{past}",
            })

        self.history.append({"role": "user", "content": user_input})
        messages += self.history

        if _needs_search(user_input):
            logger.info(f"키워드 감지! 자동 검색 시도: {user_input}")
            try:
                result = execute_tool("search_web", {"query": user_input})
                logger.info(f"검색 성공!")
                messages.append({
                    "role": "user",
                    "content": (
                        f"[실시간 검색 결과]\n{result}\n\n"
                        "위 검색 결과를 참고해서 답변하세요."
                    ),
                })
            except Exception as e:
                logger.warning(f"검색 실패 (오프라인 상태일 수 있음): {e}")

        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                full_response += delta
                yield delta

        # 스트리밍 완료 후 메모리 저장
        final = _strip_tool_calls(full_response)
        self.history.append({"role": "assistant", "content": final})
        self.memory.save(user_input, final)

    def clear_history(self):
        self.history.clear()
