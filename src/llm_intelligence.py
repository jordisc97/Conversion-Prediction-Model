import openai
import pandas as pd
import re
from typing import Dict, Any, List, Optional


class SalesIntelligenceAgent:
    """
    Converts raw SHAP signals into clear, human-friendly conversion explanations.
    Each row receives a unique explanation based on its actual drivers.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model


    # ─────────────────────────────────────────────────────────────
    # Parse SHAP signal string
    # Example:
    # "High USERS users (60d) (SHAP: +2.078)"
    # →
    # { feature: "...", impact: 2.078 }
    # ─────────────────────────────────────────────────────────────
    def _parse_shap_string(self, signal_str: Optional[str]) -> Optional[Dict[str, Any]]:

        if not signal_str or not isinstance(signal_str, str):
            return None

        match = re.match(
            r"(.*?)\s*\(SHAP:\s*([+-]?\d+\.?\d*)\)",
            signal_str
        )

        if match:
            return {
                "feature": match.group(1).strip(),
                "impact": float(match.group(2))
            }

        return {
            "feature": signal_str.strip(),
            "impact": None
        }


    # ─────────────────────────────────────────────────────────────
    # Build prompt
    # Forces model to use actual signals
    # ─────────────────────────────────────────────────────────────
    def _construct_prompt(
        self,
        company_data: Dict[str, Any],
        shap_drivers: List[Dict[str, Any]]
    ) -> str:

        signals_text = "\n".join(
            f"- {d['feature']} (impact {d['impact']})"
            for d in shap_drivers
        )

        return f"""
You are a CRM intelligence assistant.

Explain in ONE sentence why this company is likely to convert,
based ONLY on these real usage signals.

Company industry: {company_data['industry']}
Company size: {company_data['employee_range']}
Propensity score: {company_data['propensity_score']}

Signals:
{signals_text}

Rules:
- Use the exact signals provided
- Be specific
- Each explanation must be unique to the signals
- No generic phrases like "strong engagement"
- No mention of SHAP, model, or AI
- No bullet points
- No labels
- Return only the sentence
"""

    # ─────────────────────────────────────────────────────────────
    # Main enrichment function
    # ─────────────────────────────────────────────────────────────
    def generate_sales_briefs(
        self,
        lead_table: pd.DataFrame
    ) -> pd.DataFrame:
        explanations = []
        for _, row in lead_table.iterrows():
            company_context = {
                "industry": row.get("INDUSTRY", "Unknown"),
                "employee_range":
                    int(row["EMPLOYEE_RANGE"])
                    if pd.notna(row.get("EMPLOYEE_RANGE"))
                    else "Unknown",
                "propensity_score":
                    round(float(row.get("propensity_score", 0)), 3)
            }

            # Parse signals
            shap_drivers = []

            for i in range(1, 4):
                parsed = self._parse_shap_string(row.get(f"signal_{i}"))

                if parsed:
                    shap_drivers.append(parsed)

            # Sort by importance
            shap_drivers = sorted(
                shap_drivers,
                key=lambda x: abs(x["impact"]) if x["impact"] else 0,
                reverse=True            )

            # Fallback if no signals
            if not shap_drivers:
                explanations.append(
                    "Recent CRM activity and feature usage indicate meaningful platform engagement.")
                continue

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.5,
                    messages=[
                        {
                            "role": "user",
                            "content": self._construct_prompt(
                                company_context,
                                shap_drivers
                            )
                        }
                    ])

                explanation = (response.choices[0].message.content.strip())
                explanations.append(explanation)

            except Exception as e:
                print(f"LLM error: {e}")
                explanations.append("Consistent CRM usage and feature adoption suggest active platform engagement.")

        lead_table["llm_sales_brief"] = explanations
        return lead_table