import json
from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException


DEFAULT_MAX_RESULTS = 3
SEARCH_TIMEOUT_SECONDS = 5


class WebSearchTool:
    """
    Search the web for current information.

    Useful for:
    - Recent news
    - Current prices
    - Technology updates
    - Research
    - General information
    """

    def get_schema(self):
        return {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Use for recent news, prices, research, "
                "and information that may have changed. Check publication dates; "
                "search time does not mean a result is current."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "default": DEFAULT_MAX_RESULTS,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:

        # Validate input
        if not isinstance(query, str) or not query.strip():
            return json.dumps(
                {"success": False, "error": "Search query cannot be empty."}
            )

        max_results = max(1, min(int(max_results), 10))

        try:
            with DDGS(timeout=SEARCH_TIMEOUT_SECONDS) as ddgs:
                results = list(ddgs.text(query.strip(), max_results=max_results))

            searched_at = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(
                timespec="seconds"
            )
            if not results:
                return json.dumps(
                    {
                        "success": True,
                        "query": query,
                        "searched_at": searched_at,
                        "results": [],
                        "message": "No results found.",
                    }
                )

            formatted_results = []

            for result in results:
                formatted_results.append(
                    {
                        "title": result.get("title"),
                        "snippet": result.get("body"),
                        "url": result.get("href"),
                        "published_at": result.get("date"),
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "searched_at": searched_at,
                    "total_results": len(formatted_results),
                    "results": formatted_results,
                },
                ensure_ascii=False,
            )

        except DDGSException as error:
            if isinstance(error, TimeoutException):
                code = "timeout"
                message = "Web search timed out."
            elif isinstance(error, RatelimitException):
                code = "rate_limit"
                message = "Web search rate limit was reached."
            else:
                code = "provider_unavailable"
                message = "Web search provider failed."
            return json.dumps(
                {
                    "success": False,
                    "code": code,
                    "error": message,
                    "error_type": type(error).__name__,
                    "details": str(error),
                },
                ensure_ascii=False,
            )
