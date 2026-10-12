"""Short Chat, Responses, image and tool-call checks for a running server."""
import argparse
import base64
import json
import os
import struct
import urllib.request
import zlib


def red_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    image = b"\x89PNG\r\n\x1a\n"
    image += chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
    image += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32))
    return "data:image/png;base64," + base64.b64encode(image + chunk(b"IEND", b"")).decode()


def post(url, path, body):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=300) as f:
        return json.load(f)


def run(url):
    rows = []
    common = dict(model="Qwen3.8-flash-next", temperature=0, reasoning_effort="none", max_tokens=64)
    chat = post(url, "/v1/chat/completions", dict(common, messages=[dict(role="user", content="What is 2 + 3? Reply with only the number.")]))
    rows.append(dict(check="chat", passed="5" in (chat["choices"][0]["message"].get("content") or "")))
    response = post(url, "/v1/responses", dict(model=common["model"], input="What is 4 + 4? Reply with only the number.",
                                             max_output_tokens=64, temperature=0, reasoning=dict(effort="none")))
    text = "".join(c.get("text", "") for o in response.get("output", []) for c in o.get("content", []))
    rows.append(dict(check="responses", passed="8" in text and response.get("status") == "completed"))
    image = post(url, "/v1/chat/completions", dict(common, messages=[dict(role="user", content=[
        dict(type="text", text="Name the dominant color. Reply in English with one word."),
        dict(type="image_url", image_url=dict(url=red_png()))])]))
    rows.append(dict(check="image", passed="red" in (image["choices"][0]["message"].get("content") or "").lower()))
    tool = dict(type="function", function=dict(name="add", description="Add two integers", parameters=dict(
        type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"])))
    result = post(url, "/v1/chat/completions", dict(common, messages=[dict(role="user", content="Use add to calculate 2 + 3.")],
        tools=[tool], tool_choice=dict(type="function", function=dict(name="add"))))
    calls = result["choices"][0]["message"].get("tool_calls") or []
    valid = len(calls) == 1 and calls[0]["function"]["name"] == "add"
    if valid:
        valid = json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}
    rows.append(dict(check="tool_call", passed=valid))
    print(json.dumps(dict(complete=all(r["passed"] for r in rows), checks=rows), indent=2))
    if not all(r["passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    run(ap.parse_args().url)
