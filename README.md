# Global Supply Chain Disruption Impact Assessment using LangChain

## Project Overview

This project demonstrates how Large Language Models (LLMs) and Prompt Engineering can be applied to a real-world supply chain disruption analysis workflow.

The system loads structured disruption data, converts it into business context, and generates Business Impact Assessment reports using multiple prompting strategies.

## Technologies Used

- Python 3.12
- pandas
- LangChain
- OpenAI-compatible API
- VS Code

## Dataset

The dataset contains 100 global supply chain disruption events with attributes such as:

- Event Type
- Supplier Tier
- Severity Level
- Shipment Delay
- Production Downtime
- Revenue at Risk
- Financial Impact
- Mitigation Action
- Business Continuity Status

## Workflow

1. Load CSV dataset using pandas
2. Select 25 sample disruption events
3. Convert DataFrame rows into LLM-readable business context
4. Generate reports using:
   - Zero-Shot Prompting
   - Few-Shot Prompting
   - Chain-of-Thought (CoT)
   - Tree-of-Thought (ToT)
5. Compare reports using LLM-as-Judge
6. Tune inference parameters:
   - Temperature
   - Max Tokens
   - Top-p
7. Generate stakeholder-specific reports for:
   - CEO
   - Operations Manager
   - Risk Officer

## Key Findings

- More complex prompts do not always produce better outputs.
- Few-Shot prompting improves style consistency but can constrain analysis.
- CoT improves structured reasoning.
- ToT helps compare alternative mitigation strategies and trade-offs.
- Temperature affects response variation rather than quality.
- Max Tokens primarily controls output length.
- Top-p affects token diversity.
- LLM-as-Judge is useful for comparison but is not definitive.
- Numerical analytics should be validated using deterministic tools such as pandas or SQL.

## Example Parameter Results

| Experiment | Setting | Word Count |
|---|---:|---:|
| Temperature | 0.0 | 328 |
| Temperature | 0.7 | 370 |
| Temperature | 1.2 | 356 |
| Max Tokens | 150 | 103 |
| Max Tokens | 400 | 273 |
| Max Tokens | 1200 | 577 |
| Top-p | 0.1 | 460 |
| Top-p | 0.5 | 443 |
| Top-p | 1.0 | 489 |

## Stakeholder Reports

| Stakeholder | Temperature | Max Tokens | Top-p | Word Count |
|---|---:|---:|---:|---:|
| CEO | 0.2 | 450 | 0.5 | 282 |
| Operations Manager | 0.5 | 900 | 0.7 | 569 |
| Risk Officer | 0.8 | 1500 | 0.9 | 620 |

## Security

API credentials are stored locally in `config.json`.

`config.json` is excluded from version control using `.gitignore`.

## Learning Outcome

This project demonstrates an end-to-end LLM workflow from structured data preparation to prompt design, evaluation, parameter tuning, and stakeholder-specific reporting.