import openai
import pandas as pd
import json
from typing import Dict, Any

class SalesIntelligenceAgent:
    """
    Integrates GAI to transform raw ML outputs into rep-facing 'Sales Briefs'.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def _construct_prompt(self, company_data: Dict[str, Any], top_features: Dict[str, Any]) -> str:
        return f"""
        You are a Sales Strategy Assistant for HubSpot. 
        Your goal is to write a 2-sentence 'Action Brief' for a Sales Rep based on a lead's propensity model data.
        
        DATA FOR ANALYSIS:
        - Company: {company_data['name']}
        - Industry: {company_data['industry']}
        - Size: {company_data['employee_range']}
        - Top Model Drivers (SHAP values): {json.dumps(top_features)}
        - Recency: {company_data['days_since_last_action']} days since last activity.

        INSTRUCTIONS:
        1. Explain WHY they are likely to convert (e.g., specific usage spikes).
        2. Suggest a specific conversation starter or product tier (e.g., 'Starter' vs 'Pro').
        3. Be professional, concise, and persuasive. Avoid robotic language.
        
        OUTPUT FORMAT:
        "Action Brief: [Your 2 sentences here]"
        """

    def generate_sales_briefs(self, lead_table: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through the top leads and enriches them with LLM-generated insights.
        """
        briefs = []
        # We only run this for the Top K to save tokens and rep time
        for _, row in lead_table.iterrows():
            # Extract relevant context for the LLM
            company_context = {
                "name": f"Portal ID {row['company_id']}", # Or actual name if joined
                "industry": row.get('industry', 'Unknown'),
                "employee_range": row.get('employee_range', 'Unknown'),
                "days_since_last_action": row.get('days_since_last_usage', 'N/A')
            }
            
            # Extract SHAP signals from your existing explainability logic
            shap_signals = {
                "signal_1": row.get('signal_1'),
                "signal_2": row.get('signal_2')
            }

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self._construct_prompt(company_context, shap_signals)}],
                    temperature=0.7
                )
                briefs.append(response.choices[0].message.content.replace("Action Brief: ", ""))
            except Exception as e:
                # Fallback if API fails
                briefs.append("High propensity lead based on usage volume. Recommend immediate discovery call.")
        
        lead_table['llm_sales_brief'] = briefs
        return lead_table
