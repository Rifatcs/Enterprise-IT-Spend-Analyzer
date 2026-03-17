# Enterprise IT Spend Analyzer

🚀 **Live Demo:** [https://finops-ai.streamlit.app](https://finops-ai.streamlit.app)

A multi-agent AI system for enterprise IT spend analysis built on the **Technology Business Management (TBM)** framework. Upload any IT spend CSV and ask questions in plain English — a team of specialist AI agents analyzes your data, identifies waste, and generates executive-ready recommendations.

---

## Demo

Load the included sample dataset (50 vendors, ~$6M annual spend) and try:

> *"What are our top cost drivers?"*
> *"Find underutilized and duplicate services"*
> *"Generate optimization recommendations with ROI estimates"*
> *"Run a full TBM analysis"*
> *"Write an executive report for the CIO"*

---

## How It Works

A central **Orchestrator** reads every chat message and routes it to the right specialist agent. Agents share findings so later agents build on earlier insights.

| Agent | Responsibility |
|---|---|
| **Intake** | Validates data quality and schema completeness |
| **Cost Analysis** | Identifies top spend drivers, anomalies, and vendor concentration |
| **TBM** | Maps spend to IT cost pools, value streams, and business capabilities |
| **Optimization** | Generates prioritized savings recommendations with ROI estimates |
| **Report** | Produces C-suite executive summaries combining all agent findings |

**Hybrid architecture** — all financial calculations run in pandas (auditable, no hallucination). The AI layer interprets results and generates strategic narratives.

---

## Getting Started

```bash
git clone https://github.com/Rifatcs/Enterprise-IT-Spend-Analyzer.git
cd Enterprise-IT-Spend-Analyzer
pip install -r requirements.txt
```

Add your API key to a `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Run the app:

```bash
python -m streamlit run app.py
```

> **Free alternative:** Sign up at [groq.com](https://groq.com) for a free API key (Llama 3.3 70B) and set `GROQ_API_KEY` in your `.env` instead.

---

## CSV Format

The app accepts any IT spend CSV — column names are auto-detected and mapped automatically. For full analysis, include:

| Column | Description |
|---|---|
| `vendor` | Vendor or supplier name |
| `service_name` | Name of the service or tool |
| `department` | Owning department |
| `cost_category` | Cloud, SaaS, Infrastructure, etc. |
| `annual_cost` | Annual spend in USD |
| `utilization_pct` | % utilization (0–100) |
| `contract_end_date` | Contract expiry for renewal alerts |

Aliases like `supplier`, `amount`, `usage`, `expiry_date` are recognized automatically.

---

## Project Structure

```
├── app.py                          # Streamlit chat interface
├── config/prompts.py               # System prompts for all agents
├── src/
│   ├── orchestrator.py             # Routing and agent coordination
│   ├── agents/                     # 5 specialist agents + base class
│   ├── analytics/deterministic.py  # Pandas analytics engine
│   └── models/schemas.py           # Data models and TBM taxonomy
└── data/sample_it_spend.csv        # 50-vendor demo dataset
```

---

## Tech Stack

Streamlit · Anthropic Claude (Opus 4.6) · pandas · Plotly · python-dotenv

---

## License

MIT
