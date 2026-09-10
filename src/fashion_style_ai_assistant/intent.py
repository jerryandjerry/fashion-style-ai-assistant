from __future__ import annotations

import re

from .schemas import ParsedIntent


COLOR_TERMS = {
    "black": ["black", "charcoal", "黑", "黑色"],
    "white": ["white", "ivory", "cream", "白", "白色", "米色"],
    "blue": ["blue", "navy", "denim", "蓝", "蓝色"],
    "red": ["red", "burgundy", "红", "红色"],
    "green": ["green", "sage", "olive", "绿", "绿色"],
    "tan": ["tan", "sand", "beige", "camel", "卡其", "棕", "驼"],
    "silver": ["silver", "metallic", "银", "银色"],
}

OCCASION_TERMS = {
    "beach": ["beach", "coastal", "resort", "海边", "海滩", "度假"],
    "office": ["office", "work", "workwear", "办公室", "上班", "通勤"],
    "interview": ["interview", "面试"],
    "date": ["date", "dinner", "romantic", "约会", "晚餐"],
    "gym": ["gym", "running", "workout", "运动", "健身", "跑步"],
    "travel": ["travel", "trip", "vacation", "旅行", "出差"],
    "wedding_guest": ["wedding", "guest", "婚礼"],
    "weekend": ["weekend", "casual", "周末", "休闲"],
}

CATEGORY_TERMS = {
    "shirt": ["shirt", "button down", "tee", "top", "衬衫", "上衣", "t恤"],
    "pants": ["pants", "trousers", "leggings", "jeans", "裤", "裤子"],
    "dress": ["dress", "裙", "连衣裙"],
    "jacket": ["jacket", "blazer", "coat", "外套", "西装"],
    "shoes": ["shoes", "sneakers", "boots", "鞋", "鞋子"],
    "bag": ["bag", "crossbody", "包"],
    "accessory": ["scarf", "earrings", "accessory", "配饰", "耳环", "围巾"],
}

STYLE_TERMS = {
    "minimal": ["minimal", "simple", "clean", "简约"],
    "classic": ["classic", "timeless", "经典"],
    "casual": ["casual", "relaxed", "休闲", "轻松"],
    "formal": ["formal", "polished", "professional", "正式", "职业"],
    "romantic": ["romantic", "soft", "pretty", "温柔", "浪漫"],
    "sporty": ["sporty", "active", "performance", "运动"],
    "street": ["street", "edgy", "街头"],
    "coastal": ["coastal", "beachy", "resort", "海边"],
}

BUDGET_PATTERNS = [
    re.compile(r"(?:under|below|less than|<=|预算不超过|低于|小于)\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|max|以内)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:dollars|usd|以内|以下)", re.I),
]


def _match_terms(query: str, mapping: dict[str, list[str]]) -> list[str]:
    q = query.lower()
    matches: list[str] = []
    for canonical, variants in mapping.items():
        if any(v.lower() in q for v in variants):
            matches.append(canonical)
    return matches


def _extract_budget(query: str) -> float | None:
    for pattern in BUDGET_PATTERNS:
        match = pattern.search(query)
        if match:
            return float(match.group(1))
    if any(word in query.lower() for word in ["cheap", "affordable", "低预算", "便宜"]):
        return 60.0
    return None


def parse_intent(query: str) -> ParsedIntent:
    """Heuristic parser used as an offline baseline.

    In a production system, replace this with a structured LLM call or a trained intent model,
    then keep this parser as a deterministic fallback.
    """

    occasion_terms = _match_terms(query, OCCASION_TERMS)
    color_terms = _match_terms(query, COLOR_TERMS)
    category_terms = _match_terms(query, CATEGORY_TERMS)
    style_terms = _match_terms(query, STYLE_TERMS)
    budget_max = _extract_budget(query)

    needs_clarification = len(occasion_terms) == 0 and len(style_terms) == 0 and len(category_terms) == 0
    clarification_question = None
    if needs_clarification:
        clarification_question = (
            "What occasion or style should I optimize for: work, beach, date night, travel, "
            "gym, or weekend casual?"
        )

    return ParsedIntent(
        raw_query=query,
        occasion_terms=occasion_terms,
        color_terms=color_terms,
        category_terms=category_terms,
        style_terms=style_terms,
        budget_max=budget_max,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
