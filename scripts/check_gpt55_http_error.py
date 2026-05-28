#!/usr/bin/env python3
from __future__ import annotations

import json, os, urllib.error, urllib.request

api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
base = (os.getenv('OPENAI_BASE_URL') or os.getenv('LLM_BASE_URL') or 'https://api.openai.com/v1').rstrip('/')
model = os.getenv('OPENAI_MODEL') or os.getenv('LLM_MODEL') or 'gpt-4o-mini'
print('configured', bool(api_key), 'base_host', base.split('//')[-1].split('/')[0], 'model', model)
body = json.dumps({
    'model': model,
    'messages': [{'role': 'user', 'content': 'Return ok'}],
    'max_tokens': 20,
    'temperature': 0,
}).encode()
req = urllib.request.Request(base + '/chat/completions', data=body, method='POST', headers={'Content-Type':'application/json','Authorization':'Bearer '+api_key})
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print('status', resp.status)
        data = resp.read().decode()[:500]
        print(data)
except urllib.error.HTTPError as e:
    print('http_error', e.code, e.reason)
    print(e.read().decode(errors='replace')[:1000])
except Exception as e:
    print('error', type(e).__name__, str(e)[:500])
