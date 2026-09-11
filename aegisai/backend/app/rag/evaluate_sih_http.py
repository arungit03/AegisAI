"""Evaluate SIH questions through the real HTTP chat API.

Example (inside the backend container)::

    python -m app.rag.evaluate_sih_http \
      --file /app/uploads/<stored-file>.pdf \
      --username admin --password admin123

The output is JSON and includes the route, retrieval evidence, prompt context
available through API citations, model answer, expected fact, and pass/fail.
Each question uses a new conversation so history cannot mask retrieval bugs.
"""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.rag.ingestion import extract_sih_records_pdf
from app.rag.routing import QueryIntent, classify_query


DEFAULT_QUESTIONS = [
    "What is SIH26012?",
    "Tell me the problem statement of SIH26012.",
    "What is the title of SIH26012?",
    "Which organization submitted SIH26012?",
    "Which category is SIH26012?",
    "What is the theme of SIH26012?",
    "What is the deadline for SIH26012?",
    "Explain SIH26012.",
    "Give me the important details of SIH26012.",
    "What technology is required for SIH26012?",
    "What technology should I use for SIH26012?",
    "Does the document specify technology for SIH26012?",
    "What is SIH26013?",
    "Compare SIH26012 and SIH26013.",
    "Find SIH problems related to AI.",
    "List SIH problem statements in the document.",
    "Does my uploaded SIH document contain SIH26012?",
    "Does the document contain SIH99999?",
    "what is sih 26012",
    "What is SIH-26012?",
    "What title belongs to Automated Urban Parcel Mapping?",
    "What is the problem statement title for SIH26012 and SIH26013?",
]

SAFE_REFUSAL = "I couldn't find sufficient information in the authorized company knowledge base to answer this accurately."
ID_RE = re.compile(r"\bsih\s*-?\s*(\d{5})\b", re.IGNORECASE)


def _request_json(base_url: str, path: str, method: str, payload: Dict[str, Any], token: Optional[str] = None) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _ids(question: str) -> List[str]:
    return list(dict.fromkeys(f"SIH{match.group(1)}".upper() for match in ID_RE.finditer(question)))


def _method(question: str, intent: QueryIntent) -> str:
    if intent == QueryIntent.DOCUMENT_AWARE:
        return "authorized PostgreSQL document query"
    if intent == QueryIntent.COMPANY:
        if _ids(question):
            return "authorized exact structured SIH lookup"
        if re.search(r"\blist\b.*\b(?:sih|problem statements?)\b", question, re.IGNORECASE):
            return "authorized structured SIH listing"
        if re.search(r"\b(?:ps|problem statement)\s+(?:number|id)\b|\bwhat\s+title\s+(?:belongs|is associated)\b", question, re.IGNORECASE):
            return "authorized lexical title lookup"
        return "authorized semantic vector search"
    return "conversation/Ollama generation"


def _expected(question: str, records: Dict[str, Dict[str, Any]], original_filename: str) -> Dict[str, Any]:
    normalized = question.lower()
    requested = _ids(question)
    if "technology" in normalized:
        return {"fact": "Technology is not specified in the uploaded catalog PDF.", "kind": "safe_refusal"}
    if "contain" in normalized and requested:
        return {"fact": "YES" if requested[0] in records else "NO", "kind": "exact_presence"}
    if "uploaded" in normalized and "document" in normalized:
        return {"fact": original_filename, "kind": "document_list"}
    if "list" in normalized and "problem statement" in normalized:
        return {"fact": "All extracted SIH records", "kind": "structured_list"}
    if re.search(r"\b(?:ps|problem statement)\s+(?:number|id)\b|\bwhat\s+title\s+(?:belongs|is associated)\b", normalized) and not requested:
        title_match = next((record for record in records.values()
                            if record["title"].lower() in normalized), None)
        if not title_match:
            content_tokens = set(re.sub(r"[^a-z0-9]+", " ", normalized).split()) - {
                "what", "which", "is", "the", "a", "an", "title", "belongs", "belong", "to", "for", "of",
            }
            title_match = next((record for record in records.values()
                                if len(content_tokens) >= 3 and content_tokens.issubset(
                                    set(re.sub(r"[^a-z0-9]+", " ", record["title"].lower()).split())
                                )), None)
        if title_match and "what title" in normalized:
            return {"fact": title_match["title"], "kind": "title_lookup"}
        if title_match:
            matched_id = next(record_id for record_id, record in records.items() if record is title_match)
            return {"fact": matched_id, "kind": "reverse_lookup"}
        return {"fact": "matching SIH identifier", "kind": "reverse_lookup"}
    if len(requested) > 1:
        return {"fact": "; ".join(f"{record_id}: {records.get(record_id, {}).get('title', 'not found')}" for record_id in requested), "kind": "multi_record"}
    if requested:
        record = records.get(requested[0])
        if not record:
            return {"fact": f"{requested[0]} is not in the authorized document.", "kind": "not_found"}
        field = next((name for word, name in (
            ("organization", "organization"), ("category", "category"),
            ("theme", "theme"), ("deadline", "deadline"),
        ) if word in normalized), None)
        return {"fact": record[field] if field else record["title"], "kind": field or "record"}
    if "find sih" in normalized:
        return {"fact": "At least one authorized SIH record related to AI", "kind": "semantic_review"}
    return {"fact": "Manual factual review required", "kind": "review"}


def _pass(question: str, answer: str, sources: List[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    answer_lower = answer.lower()
    source_text = " ".join(source.get("chunk_text", "") for source in sources).lower()
    source_ids = set(re.findall(r"\bSIH\d{5}\b", source_text.upper()))
    kind = expected["kind"]
    normalized_answer = re.sub(r"[^a-z0-9]+", " ", answer_lower).strip()
    normalized_fact = re.sub(r"[^a-z0-9]+", " ", str(expected["fact"]).lower()).strip()
    if kind == "safe_refusal":
        return SAFE_REFUSAL.lower() in answer_lower
    if kind == "exact_presence":
        return answer.strip().upper() == expected["fact"] or answer_lower.startswith(expected["fact"].lower())
    if kind == "document_list":
        return expected["fact"].lower() in answer_lower
    if kind == "structured_list":
        return bool(sources) and "SIH" in answer
    if kind == "not_found":
        return "couldn't find" in answer_lower and not sources
    if kind == "reverse_lookup":
        return answer.strip().upper() == expected["fact"]
    if kind == "title_lookup":
        return expected["fact"].lower() in answer_lower and bool(sources)
    if kind == "multi_record":
        requested = _ids(question)
        return all(identifier in source_ids for identifier in requested)
    if kind in {"organization", "category", "theme", "deadline"}:
        return normalized_fact in normalized_answer and bool(sources)
    if kind == "semantic_review":
        return bool(sources) and SAFE_REFUSAL.lower() not in answer_lower
    if kind == "record":
        return normalized_fact in normalized_answer and bool(sources)
    return False


def evaluate(base_url: str, username: str, password: str, file_path: str, questions: List[str]) -> List[Dict[str, Any]]:
    token = _request_json(base_url, "/api/auth/login", "POST", {"username": username, "password": password})["access_token"]
    parsed = {record["problem_statement_id"]: record for record in extract_sih_records_pdf(file_path)}
    original_filename = "PS SIH 26 -1.pdf"
    results = []
    for question in questions:
        intent = classify_query(question)
        expected = _expected(question, parsed, original_filename)
        try:
            response = _request_json(base_url, "/api/chat", "POST", {"message": question}, token)
            sources = response.get("sources", [])
            results.append({
                "question": question,
                "route_selected": intent.value,
                "retrieval_method": _method(question, intent),
                "retrieved_document": sorted({source.get("document_filename") for source in sources}),
                "retrieved_page": sorted({source.get("page_number") for source in sources if source.get("page_number") is not None}),
                "retrieved_score": [source.get("similarity_score") for source in sources],
                "context": [source.get("chunk_text", "") for source in sources],
                "model_answer": response.get("message", ""),
                "expected_fact": expected["fact"],
                "pass": _pass(question, response.get("message", ""), sources, expected),
                "http_status": 200,
            })
        except (HTTPError, URLError, TimeoutError, KeyError) as error:
            results.append({
                "question": question,
                "route_selected": intent.value,
                "retrieval_method": _method(question, intent),
                "retrieved_document": [], "retrieved_page": [], "retrieved_score": [], "context": [],
                "model_answer": f"HTTP evaluation error: {type(error).__name__}",
                "expected_fact": expected["fact"], "pass": False,
                "http_status": getattr(error, "code", None),
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    results = evaluate(args.base_url, args.username, args.password, args.file, DEFAULT_QUESTIONS)
    print(json.dumps({"count": len(results), "passed": sum(item["pass"] for item in results), "failed": sum(not item["pass"] for item in results), "results": results}, indent=2))
    return 0 if all(item["pass"] for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
