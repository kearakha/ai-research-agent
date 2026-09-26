# AI Research Agent

Belajar fundamental agent: goal → LLM decision → tool call → observation →
state → loop → stop. Bukan platform riset production.

## Cara kerja

```
goal → Planner (pecah jadi 2-4 subtopik)
     → Researcher × N subtopik (search_web / read_page, loop sampai "final")
     → merge jawaban tiap subtopik
     → Critic (cek jawaban gabungan sudah nutup goal atau belum)
     → kalau "needs_more": 1 ronde riset tambahan buat gap itu, lalu Critic lagi
     → catat goal + jawaban ke memory.jsonl
     → report.json + report.md
```

Pipeline dibatasi maksimal 1 ronde riset tambahan (nggak di-loop sampai Critic
bilang "ok" terus-terusan) dan maksimal 3 subtopik — biar nggak gampang kena
rate limit provider LLM free tier.

Tiap Researcher jalanin loop eksplisit yang sama: LLM balas satu action JSON
(`search_web`, `read_page`, atau `final`) per giliran, hasil tool di-feed
balik sebagai observation, ulang sampai `final` atau limit iterasi/tool call
kena (lalu dipaksa rangkum).

Tool calling pakai prompt biasa (bukan `tools` param OpenAI-style) karena
model gratisan yang dipakai selama development hang/balas kosong kalau
`tools` disertakan — endpoint completion polos + instruksi format JSON di
system prompt jauh lebih stabil.

Memory (`memory.py`) nyimpen tiap run ke `memory.jsonl` dan nyari run lama
yang mirip lewat keyword overlap (bukan embedding) — kalau ketemu, jawaban
lama itu di-suntik sebagai konteks ke subtopik pertama.

## Setup

```
pip install -r requirements.txt
cp .env.example .env
```

Isi `.env`:
- `TOKENROUTER_API_KEY` / `TOKENROUTER_BASE_URL` / `TOKENROUTER_MODEL` — LLM.
  Bisa TokenRouter (`z-ai/glm-5.3-free`) atau endpoint OpenAI-compatible lain
  (mis. Gemini) selama formatnya sama.
- `TAVILY_API_KEY` — buat `search_web` (tavily.com, free tier 1.000
  credits/bulan, no card).

## Jalanin

```
python agent.py "<goal riset>"
```

Output: `report.json` (data lengkap) + `report.md` (ringkasan baca).
Log progres (`[AGENT]`, `[PLANNER]`, `[RESEARCHER]`, `[CRITIC]`, `[MEMORY]`,
`[TOOL]`) ke stderr.

## Test

```
python test_agent.py
python test_memory.py
```

Semua test pakai fake LLM / temp file — nggak ada network call, nggak butuh
API key.
