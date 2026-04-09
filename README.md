# UCLA Course Planner — Multi-Agent Pipeline

An 8-agent pipeline built on [Fetch.ai uAgents](https://fetch.ai/docs/guides/agents/getting-started) and [ASI:One](https://asi1.ai/) that collects a UCLA student's profile through conversation, scrapes live course data, and generates ranked schedule recommendations with a full Markdown report.

## Architecture

```
User ←→ input_agent (8001)
              ↓
      available_classes_agent (8002)
              ↓
        enrollment_agent (8003)
            ↓          ↓
    bruinwalk_agent  grade_dist_agent
       (8004)           (8005)
            ↓          ↓
        schedule_agent (8006)   ← waits for both
              ↓
        ranking_agent (8007)
              ↓
        report_agent (8008) → sends report back to user
```

## Setup

### 1. Install dependencies

```bash
pip install uagents openai httpx beautifulsoup4 pdfplumber dacite pandas playwright
```

Optional (for Bruinwalk SPA fallback rendering):

```bash
playwright install chromium
```

### 2. Get an ASI:One API key

1. Go to [https://asi1.ai/](https://asi1.ai/)
2. Sign up and create an API key
3. Export it:

```bash
export ASI1_API_KEY="your-key-here"
```

### 3. Agent addresses

All 8 agent addresses are already populated in `course_planner/utils.py` under `AGENT_ADDRESSES`. These were generated from each agent's seed phrase on first run.

If you change any seed phrase, you'll need to:

1. Run that agent individually: `python -m course_planner.agents.<agent_name>`
2. Copy the printed address from the startup log
3. Update the corresponding entry in `AGENT_ADDRESSES`

Current addresses:

| Agent | Port | Address |
|-------|------|---------|
| input | 8001 | `agent1qvzm8976...` |
| available_classes | 8002 | `agent1qdq95wua...` |
| enrollment | 8003 | `agent1q084cz3f...` |
| bruinwalk | 8004 | `agent1qwacpk27...` |
| grade_dist | 8005 | `agent1qwy2uv9h...` |
| schedule | 8006 | `agent1q00ey3tr...` |
| ranking | 8007 | `agent1qvh2j5fl...` |
| report | 8008 | `agent1qdnzpjfw...` |

## Running

### All agents at once

```bash
cd ~/Agents\ 2
python run_all.py
```

This starts all 8 agents as separate processes. Press `Ctrl+C` to stop all. Crashed agents are automatically restarted.

### Individual agents

```bash
cd ~/Agents\ 2
python -m course_planner.agents.input_agent
python -m course_planner.agents.available_classes_agent
python -m course_planner.agents.enrollment_agent
python -m course_planner.agents.bruinwalk_agent
python -m course_planner.agents.grade_dist_agent
python -m course_planner.agents.schedule_agent
python -m course_planner.agents.ranking_agent
python -m course_planner.agents.report_agent
```

### Orchestrator shortcut

```bash
python -m course_planner.agent
```

This runs only the input agent (the user-facing entry point). The other 7 agents must be running separately.

## How it works

1. **input_agent** — Multi-turn conversation collecting: name, major, year, GPA, units, enrollment pass, DARS report, required/preferred courses, scheduling constraints, format preference, unit range
2. **available_classes_agent** — Scrapes UCLA Schedule of Classes, filters by prerequisites and format preference
3. **enrollment_agent** — Scrapes historical enrollment data, predicts open-seat probability per pass
4. **bruinwalk_agent** — Scrapes Bruinwalk for professor and course ratings, computes composite scores
5. **grade_dist_agent** — Loads UCLA grade distribution data from 4 Google Sheets (6,000+ courses)
6. **schedule_agent** — Merges bruinwalk + grade data, generates conflict-free schedule combinations, scores by quality
7. **ranking_agent** — Computes weighted composite scores, validates constraints, ranks top 3 with LLM-generated explanations
8. **report_agent** — Builds a full Markdown report with risk flags, fallback suggestions, and methodology

## File structure

```
Agents 2/
├── run_all.py                          ← multi-process launcher
├── README.md
└── course_planner/
    ├── __init__.py
    ├── agent.py                        ← orchestrator entry point
    ├── utils.py                        ← dataclasses, serialize/deserialize, addresses
    ├── agents/
    │   ├── __init__.py
    │   ├── input_agent.py              ← port 8001
    │   ├── available_classes_agent.py  ← port 8002
    │   ├── enrollment_agent.py         ← port 8003
    │   ├── bruinwalk_agent.py          ← port 8004
    │   ├── grade_dist_agent.py         ← port 8005
    │   ├── schedule_agent.py           ← port 8006
    │   ├── ranking_agent.py            ← port 8007
    │   └── report_agent.py             ← port 8008
    └── scrapers/
        ├── __init__.py
        ├── soc_scraper.py              ← UCLA Schedule of Classes
        ├── bruinwalk_scraper.py        ← Bruinwalk ratings
        └── grade_dist_scraper.py       ← Google Sheets grade data
```
