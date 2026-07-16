"""Reusable teacher-API client with an automatic free-first provider fallback chain.

Any script needing a teacher model (label distillation, relabeling, mining) should use this rather
than hand-rolling provider logic. It encodes the operational policy in code so it is always applied:

    Ollama Cloud  ->  Cloudflare Workers AI  ->  Ollama Cloud (after its ~2h session reset)
                  ->  OpenRouter (LAST RESORT: spends real credits)

Behaviour it handles for you:
  * Credentials: env first, else the `o` password manager (never hardcoded).
  * Browser User-Agent — Cloudflare-fronted hosts (Ollama, Groq) 403 the default python-urllib UA
    at the edge, which silently looks like a hang/quota problem.
  * Exhaustion detection ("session usage limit", "daily free allocation"/neurons, "insufficient
    credits") -> mark provider exhausted, fall through to the next one.
  * 429s: honour Retry-After for short cooldowns; treat long ones as exhaustion and move on.
  * Ollama's session cap resets in ~2h, so an exhausted Ollama becomes eligible again after
    OLLAMA_RESET_S and the chain will cycle back to it before touching OpenRouter.

Usage:
    from prod_memory.teacher_client import TeacherClient
    tc = TeacherClient()
    content = tc.complete(system="...", user="...", max_tokens=220)   # None if all exhausted
    print(tc.stats)   # {'ollama': 812, 'cloudflare': 134, ...}
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXHAUSTED_RE = re.compile(
    r"session usage limit|daily free allocation|10,?000 neurons|out of credits|"
    r"insufficient credits|upgrade for higher limits|quota exceeded", re.I)

OLLAMA_RESET_S = float(os.environ.get("OLLAMA_RESET_S", 2 * 3600))  # session cap resets ~2h


def _secret(key: str) -> str:
    """Fetch a secret: env override first, else `o <key> -r` (returns on stdout).

    Use `-r`, never the bare `o <key>` + clipboard read: the clipboard round-trip HANGS inside
    detached/background processes (it cost us a silent multi-hour stall), and it races anything else
    touching the clipboard. `-r` writes straight to stdout, so it works headless.
    """
    env = os.environ.get(key.upper())
    if env:
        return env.strip()
    try:
        p = subprocess.run(["o", key, "-r"], capture_output=True, text=True,
                           check=False, timeout=30)
        return (p.stdout or "").strip()
    except Exception:
        return ""


class _Provider:
    def __init__(self, name, url, model, key, paid=False, reset_s=None):
        self.name, self.url, self.model, self.key = name, url, model, key
        self.paid = paid
        self.reset_s = reset_s           # if set, provider becomes eligible again after this
        self.exhausted_at = None

    @property
    def available(self) -> bool:
        if not self.key or not self.url:
            return False
        if self.exhausted_at is None:
            return True
        if self.reset_s is None:
            return False
        return (time.time() - self.exhausted_at) >= self.reset_s

    def mark_exhausted(self):
        self.exhausted_at = time.time()


def default_chain() -> list[_Provider]:
    ol_key = _secret("ollamakey")
    cf_acct = _secret("cloudflareaccountid")
    cf_key = _secret("cloudflarekey")
    or_key = _secret("openrouterkey")
    chain = [
        # free, fast, session cap resets ~2h -> eligible again later, so it is retried before paid
        _Provider("ollama", "https://ollama.com/v1/chat/completions",
                  "qwen3-coder-next", ol_key, reset_s=OLLAMA_RESET_S),
        # free, 10k neurons/day
        _Provider("cloudflare",
                  f"https://api.cloudflare.com/client/v4/accounts/{cf_acct}/ai/v1/chat/completions",
                  "@cf/meta/llama-3.3-70b-instruct-fp8-fast", cf_key),
        # last resort: real money
        _Provider("openrouter", "https://openrouter.ai/api/v1/chat/completions",
                  "qwen/qwen3-coder-next", or_key, paid=True),
    ]
    return [p for p in chain if p.key]


class AllProvidersExhausted(RuntimeError):
    pass


class TeacherClient:
    def __init__(self, providers=None, allow_paid=True, min_interval_s=0.0, verbose=True):
        self.providers = providers if providers is not None else default_chain()
        self.allow_paid = allow_paid
        self.min_interval_s = min_interval_s
        self.verbose = verbose
        self.stats = {}
        self._last_call = 0.0

    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _pick(self):
        """First available provider, preferring free ones; paid only if nothing free is available."""
        for p in self.providers:
            if p.paid:
                continue
            if p.available:
                return p
        if self.allow_paid:
            for p in self.providers:
                if p.paid and p.available:
                    return p
        return None

    def _request(self, prov, system, user, max_tokens, temperature, timeout):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        body = json.dumps({"model": prov.model, "messages": msgs,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(prov.url, data=body, headers={
            "Authorization": f"Bearer {prov.key}", "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    def complete(self, user, system=None, max_tokens=256, temperature=0.0, timeout=90, retries=3):
        """Return the completion text, transparently falling through the provider chain.

        Returns None on a non-quota failure for this item. Raises AllProvidersExhausted when every
        provider (including any eligible-again Ollama) is out.
        """
        while True:
            prov = self._pick()
            if prov is None:
                waits = [p for p in self.providers if p.reset_s and p.exhausted_at]
                raise AllProvidersExhausted(
                    "all providers exhausted" + (f"; ollama eligible again in "
                    f"{max(0, int(min((p.reset_s - (time.time()-p.exhausted_at)) for p in waits)))}s"
                    if waits else ""))
            for attempt in range(retries + 1):
                if self.min_interval_s:
                    gap = self.min_interval_s - (time.time() - self._last_call)
                    if gap > 0:
                        time.sleep(gap)
                try:
                    out = self._request(prov, system, user, max_tokens, temperature, timeout)
                    self._last_call = time.time()
                    self.stats[prov.name] = self.stats.get(prov.name, 0) + 1
                    return out
                except urllib.error.HTTPError as e:
                    self._last_call = time.time()
                    txt = ""
                    try:
                        txt = e.read().decode("utf-8", "ignore")
                    except Exception:
                        pass
                    if EXHAUSTED_RE.search(txt or ""):
                        self._log(f"[teacher] {prov.name} exhausted -> next provider")
                        prov.mark_exhausted()
                        break  # re-pick
                    if e.code == 429:
                        ra = e.headers.get("retry-after")
                        w = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 5 * 2 ** attempt
                        if w > 240:
                            self._log(f"[teacher] {prov.name} long cooldown {w:.0f}s -> next provider")
                            prov.mark_exhausted()
                            break
                        time.sleep(min(w, 240))
                        continue
                    if e.code in (401, 403):
                        self._log(f"[teacher] {prov.name} auth/access {e.code} -> disabling")
                        prov.mark_exhausted()
                        prov.reset_s = None
                        break
                    if e.code in (404, 410):
                        # Model retired/renamed (Ollama pulled qwen3-coder-next on 2026-07-14 -> 410
                        # Gone). This is permanent for this provider+model, so retrying per item just
                        # grinds. Kill the provider outright and fall through.
                        self._log(f"[teacher] {prov.name} model '{prov.model}' gone ({e.code}) "
                                  f"-> disabling provider")
                        prov.mark_exhausted()
                        prov.reset_s = None
                        break
                    if 400 <= e.code < 500:
                        # Other persistent client errors: don't burn the whole run on one provider.
                        prov.consec_4xx = getattr(prov, "consec_4xx", 0) + 1
                        if prov.consec_4xx >= 5:
                            self._log(f"[teacher] {prov.name} {prov.consec_4xx} consecutive {e.code}s "
                                      f"-> disabling provider")
                            prov.mark_exhausted()
                            prov.reset_s = None
                            break
                    if attempt >= retries:
                        return None
                    time.sleep(2 * 2 ** attempt)
                except Exception:
                    self._last_call = time.time()
                    if attempt >= retries:
                        return None
                    time.sleep(2 * 2 ** attempt)
            # provider was exhausted mid-loop -> outer while re-picks


def complete_json(tc: TeacherClient, user, system=None, **kw):
    """complete() + tolerant JSON extraction. Returns dict or None."""
    out = tc.complete(user, system=system, **kw)
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        s, e = out.find("{"), out.rfind("}")
        if s < 0 or e <= s:
            return None
        try:
            return json.loads(out[s:e + 1])
        except Exception:
            return None
