# Enterprise IT Spend Analyzer

A production-grade multi-agent AI system for enterprise IT spend analysis, built on the **Technology Business Management (TBM)** framework. Designed as an Apptio-style solution that gives IT and Finance leaders AI-powered insights into their technology portfolio.

---

## What It Does

Upload any IT spend CSV and ask questions in plain English. A team of specialist AI agents work together to analyze your data, identify waste, map spend to business value, and generate executive-ready recommendations.

**Example questions you can ask:**
- *"How much did we spend this year and where did it go?"*
- *"Find all underutilized and duplicate services"*
- *"What are our top optimization opportunities with ROI estimates?"*
- *"Map our spend to TBM cost pools and value streams"*
- *"Generate a full executive report for the CIO"*

---

## Architecture

The system uses a **chat-first multi-agent architecture** where a central Orchestrator routes every question to the right specialist agent.

```
User Question
     │
     ▼
┌─────────────┐
│ Orchestrator │  ← Routes by keyword + LLM fallback
└──────┬──────┘
       │
  ┌────┴────────────────────────────────────┐
  │                                         │
  ▼                                         ▼
Intake Agent          Cost Analysis Agent
  │                         │
  ▼                         ▼
TBM Agent            Optimization Agent
  │                         │
  └──────────┬──────────────┘
             ▼
        Report Agent
```

### The 5 Specialist Agents

| Agent | Responsibility |
|---|---|
| **Intake Agent** | Validates data quality, detects schema issues, scores completeness |
| **Cost Analysis Agent** | Identifies top spend drivers, anomalies, vendor concentration, trends |
| **TBM Agent** | Maps spend to IT cost pools, value streams, and business capabilities |
| **Optimization Agent** | Generates prioritized savings recommendations with ROI estimates |
| **Report Agent** | Produces C-suite executive reports combining all agent insights |

### Hybrid Architecture

- **Deterministic Layer** (pandas) — All financial calculations, aggregations, anomaly detection, and utilization analysis are computed in code. Numbers are always auditable and reproducible — no LLM hallucination of figures.
- **AI Layer** (Claude/LLM) — Interprets the computed analytics, provides strategic context, generates narratives, and synthesizes cross-agent insights.

---

## Key Features

- **Chat-first interface** — Ask any question in natural language; the orchestrator routes it automatically
- **Multi-agent collaboration** — Agents share findings so later agents build on earlier insights
- **TBM-aligned analysis** — Spend mapped to IT cost pools (Infrastructure, Applications, Management) and value streams (Revenue Generation, Innovation Enablement, Risk Mitigation, Operational Efficiency)
- **Flexible CSV ingestion** — Accepts any IT spend CSV format; auto-detects and maps column names
- **Inline visualizations** — Charts and data tables rendered directly in the chat
- **Contract renewal alerts** — Flags contracts expiring within 180 days for renegotiation leverage
- **Waste detection** — Identifies underutilized services, duplicate tools, and cost anomalies
- **Export** — Download the enriched dataset as CSV for Excel / Power BI / Tableau

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| AI / LLM | Anthropic Claude (or Groq/Llama free tier) |
| Data Processing | pandas, numpy |
| Visualizations | Plotly Express |
| Environment | python-dotenv |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Rifatcs/Enterprise-IT-Spend-Analyzer.git
cd Enterprise-IT-Spend-Analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Get your API key from [console.anthropic.com](https://console.anthropic.com). Requires a paid account with credits.

> **Free alternative:** The app also supports Groq's free tier (Llama 3.3 70B). See [Using Groq (Free)](#using-groq-free) below.

### 4. Run the app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the App

1. **Load data** — Upload your IT spend CSV using the sidebar, or click **Load Sample Data** to explore with the included 50-vendor demo dataset
2. **Ask questions** — Type any question in the chat input at the bottom
3. **Use Quick Actions** — Sidebar buttons for common analyses (Cost Analysis, Find Waste, TBM Mapping, etc.)
4. **Export** — Download the enriched CSV from the sidebar

### CSV Format

The app accepts any IT spend CSV. For best results, include columns for:

| Column | Description | Required |
|---|---|---|
| `vendor` | Vendor / supplier name | Yes |
| `service_name` | Name of the service or tool | Yes |
| `department` | Owning department | Yes |
| `cost_category` | Category (Cloud, SaaS, Infrastructure, etc.) | Yes |
| `annual_cost` | Annual spend in USD | Yes |
| `utilization_pct` | % utilization (0-100) | Recommended |
| `contract_end_date` | Contract expiry date | Recommended |
| `monthly_cost` | Monthly spend (derived if missing) | Optional |
| `spend_type` | CAPEX or OPEX | Optional |
| `headcount_supported` | Number of users supported | Optional |

Column names are auto-detected — `supplier`, `amount`, `usage`, `expiry_date`, etc. are all recognized automatically.

---

## Using Groq (Free)

[Groq](https://groq.com) offers a free API tier with Llama 3.3 70B — a capable open-source model.

1. Sign up at [groq.com](https://groq.com) and create an API key
2. Install the Groq SDK: `pip install groq`
3. Update your `.env`:

```env
GROQ_API_KEY=gsk_your-key-here
```

---

## Project Structure

```
enterprise_it_spend_analyzer/
├── app.py                          # Main Streamlit application
├── config/
│   └── prompts.py                  # System prompts for all agents
├── src/
│   ├── orchestrator.py             # Central routing and coordination
│   ├── agents/
│   │   ├── base_agent.py           # Base class for all agents
│   │   ├── intake_agent.py         # Data quality assessment
│   │   ├── cost_analysis_agent.py  # Cost driver analysis
│   │   ├── tbm_agent.py            # TBM framework mapping
│   │   ├── optimization_agent.py   # Savings recommendations
│   │   └── report_agent.py         # Executive report generation
│   ├── analytics/
│   │   └── deterministic.py        # Pandas-based analytics engine
│   └── models/
│       └── schemas.py              # Data models and TBM taxonomy
├── data/
│   └── sample_it_spend.csv         # 50-vendor demo dataset
├── .env                            # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## Sample Data

The included `sample_it_spend.csv` contains a realistic 50-service enterprise IT portfolio including:
- Major cloud providers (AWS, Azure, Google Cloud)
- Enterprise SaaS (Salesforce, SAP, Workday, ServiceNow)
- Security tools (Okta, CrowdStrike, Palo Alto, Zscaler)
- Collaboration tools (Microsoft 365, Slack, Zoom)
- Known issues pre-seeded: underutilized VMs, duplicate BI tools, overlapping collaboration suites, legacy mainframe, upcoming renewals

---

## TBM Framework

This system implements the **Technology Business Management (TBM)** framework — the industry standard for IT financial management used by Apptio and adopted by enterprise CIOs worldwide.

**TBM Cost Pools:**
- **IT Infrastructure** — Cloud, on-premise servers, networking, telecom
- **IT Applications** — SaaS, software licenses, custom applications
- **IT Management** — Professional services, security, governance

**TBM Value Streams:**
- **Revenue Generation** — Sales, Marketing, Customer Success
- **Innovation Enablement** — Engineering, Data & Analytics
- **Risk Mitigation** — IT Security, IT Operations
- **Operational Efficiency** — Finance, HR, Legal, Corporate

---

## License

MIT License — free to use, modify, and distribute.
