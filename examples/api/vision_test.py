import os
import base64, json, sys, time, urllib.request, urllib.error
key = open(os.path.expanduser("~/ps5lab/.key")).read().strip()
def ask(image_path, question, max_tokens=80):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode()}},
               {"type": "text", "text": question}]
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}
    r = urllib.request.Request("http://ps5.tailnet.example:8081/v1/chat/completions", json.dumps(body).encode(),
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=300))
    except urllib.error.HTTPError as e:
        return "HTTP %d %s" % (e.code, e.read().decode()[:300])
    return "%.1fs: %s" % (time.time() - t, d["choices"][0]["message"]["content"].strip())
if __name__ == "__main__":
    print(ask(sys.argv[1], sys.argv[2]))
