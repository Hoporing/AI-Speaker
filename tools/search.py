from ddgs import DDGS


def search_web(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:

            # 1. 특정 사이트 우선 검색
            priority_query = f"{query} (site:namu.wiki OR site:news.naver.com OR site:youtube.com)"
            results = list(ddgs.text(priority_query, max_results=max_results))

            # 2. 결과가 없으면 전체 웹 검색으로 전환
            if not results:
                results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "검색 결과가 없습니다."
        return "\n\n".join(
            f"제목: {r['title']}\n내용: {r['body']}" for r in results
        )
    except Exception as e:
        return f"검색 실패: {str(e)}"
