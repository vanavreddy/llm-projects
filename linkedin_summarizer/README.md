
## LinkedIn Email Agent

### Overview

An intelligent agent that processes LinkedIn job alert emails, extracts job listings, and can be extended to rate jobs against your profile. Solves the real problem of manually scanning dozens of daily job emails.

### Installation & Setup

```bash
# 1. Enable Gmail API
# - Go to https://console.cloud.google.com/
# - Create project: "Gmail Agent"
# - Enable Gmail API
# - Create OAuth 2.0 Desktop credentials
# - Download credentials.json

# 2. Install dependencies
cd agent_summarizer
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install langchain-ollama

# 3. Place credentials.json in agent_summarizer/
```

### Usage

**First Run (Authorization):**
```bash
python gmail_agent.py

# Browser opens → Sign in to Gmail → Grant permissions
# token.json created (saved for future runs)
```

**Subsequent Runs:**
```bash
python gmail_agent.py  # No browser needed
```

**Example Output:**
```
Found 5 LinkedIn emails

Processing: "lead software architect": Jobgether - Senior...
  ✅ Extracted 6 jobs

--- Email 1 ---
Date: Wed, 18 Feb 2026 13:49:42 +0000 (UTC)
Subject: "lead software architect": Jobgether - Senior Distributed Systems Engineer
Type: jobs

Content:
Found 6 job(s):

1. Senior Distributed Systems Engineer (Remote)
   Company: Jobgether
   Location: United States

2. Technical Fellow
   Company: AHEAD
   Location: United States
   Salary: $300K-$400K / year

[...]
```

### Design Decisions & Learnings

**Pattern Matching vs LLM:**
- Used pattern matching for extraction (faster, more reliable)
- Reserved LLM for classification and future matching
- LinkedIn's format is consistent enough for regex

**Classification Strategy:**
- Started with strict keyword matching
- Expanded to inclusive classification (quotes in subject = job alert)
- Fallback: try extraction even if classified as "other"

### Future Enhancements (Job Matching Extension)

**Planned Architecture:**
```
Job Extraction (Current)
    ↓
Job Description Retrieval
  - Option A: LinkedIn API (requires approval)
  - Option B: Web scraping service (Apify, ScrapingBee)
  - Option C: Manual paste
    ↓
LLM-Based Matching
  - Compare job description to resume/profile
  - Generate 1-10 match score
  - Identify strengths and gaps
    ↓
Filtered Results (only high-scoring matches)
```

### File Structure

```
agent_summarizer/
├── gmail_agent_fixed.py          # Working extraction agent
├── gmail_agent_with_matching.py  # Extension with job rating
├── credentials.json              # OAuth credentials (not in git)
├── token.json                    # Auth token (not in git)
└── README.md
```

### Security Notes

- `credentials.json`: OAuth app credentials (can share)
- `token.json`: YOUR Gmail access token (NEVER share or commit)

---
## Acknowledgments

- Kaggle for the USA Real Estate Dataset
- Ollama for making local LLM inference accessible
- LangChain for RAG abstractions
- Google for Gmail API
