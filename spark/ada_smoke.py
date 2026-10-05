"""Send 3 known requests to a running clef_server.py and print each answer, to compare with the Spark's (spark/ada_validate.md).

    python spark/ada_smoke.py [--url http://127.0.0.1:8031] [--json]

Standard library only. The first request is spark/req1.json.
"""
import argparse
import json
import time
import urllib.request

REQUESTS = {
    "req1": {"model": "kev-latest", "state": "I was charged twice. Please refund the duplicate payment today.", "questions": {
        "billing": {"type": "noul", "instructions": "Is this a billing issue?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments", "shipping": "Deliveries", "accounts": "Account access"}},
        "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Urgent", "Emergency"]}}},
    "req2": {"model": "kev-latest", "state": {"order": "A-1043", "carrier": "UPS", "status": "delivered", "customer_note":
             "The box arrived crushed and the lamp inside is broken. I want a replacement, not a refund."}, "questions": {
        "damaged": {"type": "noul", "instructions": "Did the item arrive damaged?"},
        "wants": {"type": "choice", "instructions": "What does the customer want?",
                  "criteria": {"refund": "Money back", "replacement": "A new item sent", "repair": "Fix the item"}},
        "sentiment": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["Calm", "Annoyed", "Angry", "Furious"]}}},
    "req3": {"model": "kev-latest", "state": "def add(a, b):\n    return a - b\n\nassert add(2, 2) == 4", "questions": {
        "passes": {"type": "noul", "instructions": "Does the assertion pass when this code runs?"},
        "bug": {"type": "choice", "instructions": "What is wrong with the function?",
                "criteria": {"wrong_operator": "It uses the wrong arithmetic operator", "syntax": "It has a syntax error",
                             "none": "Nothing is wrong"}}}},
}


def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def summary(answer):
    if answer["type"] == "noul": return f"noul {answer['noul']:.4f}"
    if answer["type"] == "choice": return f"choice {answer['choice']} ({answer['confidence']:.4f})"
    return f"score {answer['score']:.4f} (confidence {answer['confidence']:.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8031")
    ap.add_argument("--json", action="store_true", help="print the full responses")
    a = ap.parse_args()
    with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
        print(json.dumps(json.loads(r.read())["models"][0]))
    for name, body in REQUESTS.items():
        t = time.perf_counter()
        resp = post(a.url, body)
        wall = 1000 * (time.perf_counter() - t)
        print(f"{name}  ({resp['usage']['input_tokens']} tokens, server {resp['latency_ms']:.0f} ms, wall {wall:.0f} ms)")
        for qid, ans in resp["answers"].items():
            print(f"  {qid:10s} {summary(ans)}")
        if a.json: print(json.dumps(resp, indent=1))


if __name__ == "__main__":
    main()
