import os
import json
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

file_name = "config.json"

with open(file_name, "r") as file:
    config = json.load(file)

OPENAI_API_KEY = config.get("API_KEY")
OPENAI_API_BASE = config.get("OPENAI_API_BASE")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

print("API credentials loaded successfully.")

MODEL_NAME = "gpt-4o-mini"

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
    temperature=0
)

#print("LLM initialized successfully.")

DATA_FILE = "supply_chain_disruptions_100_events.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
#print("Shape:", df.shape)

#print("\nColumn names:")
#print(df.columns.tolist())

#print("\nFirst 5 rows:")
#print(df.head())

SAMPLE_SIZE = 25
sample_df = df.head(SAMPLE_SIZE)

#print("\nSample shape:", sample_df.shape)

business_context = ""

for index, row in sample_df.iterrows():
    business_context += f"\n========== Supply Chain Disruption Event {index + 1} ==========\n"

    for column in sample_df.columns:
        business_context += f"{column.replace('_', ' ').title()}: {row[column]}\n"

    business_context += "\n"

print("\nBusiness context preview:")
print(business_context[:500] + "...")

#zero_shot_system_prompt = """
#You are a Senior Supply Chain Risk Analyst at TransGlobal Manufacturing.

#### Expected Behavior

#- Act as an experienced supply chain risk analyst.
#- Base all observations, conclusions, and recommendations solely on the information provided in the input.
#- Do not assume, infer, or fabricate facts that are not explicitly supported by the dataset.
#- If sufficient evidence is unavailable, clearly indicate that additional operational data is required.
#- Maintain a professional, objective, and business-oriented tone.
#- Focus on producing accurate, concise, and actionable insights that support executive decision-making.

## Output Requirements

#- Present the report using clear section headings.
#- Use bullet points wherever appropriate.
#- Do not generate a report template.
#- Do not include placeholder fields such as Date, Prepared By, Company Name, or Contact Information.
#"""

#zero_shot_user_prompt = PromptTemplate.from_template("""
#Analyze the following supply chain disruption dataset and prepare a Business Impact Assessment Report.

## Input Data

#{business_context}

## Report Structure

#1. Executive Summary
#2. Identified Disruptions
#3. Root Causes
#4. Supplier Impact
#5. Inventory Impact
#6. Production Risk
#7. Logistics Risk
#8. Financial Implications
#9. Recommended Mitigation Strategies
#10. Business Continuity Assessment
#""")

#zero_shot_prompt_text = zero_shot_user_prompt.format(
    #business_context=business_context
#)

# response_zero_shot = llm.invoke([
#    SystemMessage(content=zero_shot_system_prompt),
#    HumanMessage(content=zero_shot_prompt_text),
#])

#print("\n===== ZERO-SHOT REPORT =====\n")
#print(response_zero_shot.content)

#few_shot_system_prompt = """
#You are a Senior Supply Chain Risk Analyst at TransGlobal Manufacturing.

### Expected Behavior

#- Act as an experienced supply chain risk analyst.
#- Analyze supply chain disruption data and generate evidence-based business impact assessment reports.
#- Learn the expected analytical style, reasoning approach, and response format from the reference examples provided by the user.
#- Apply the learned style consistently while analyzing the user's input.
#- Base all observations, conclusions, and recommendations solely on the information provided.
#- Do not assume, infer, or fabricate facts that are not explicitly supported by the input.
#- If sufficient evidence is unavailable, clearly indicate that additional operational data is required.
#- Maintain a professional, objective, and business-oriented tone.
#- Produce concise, accurate, and actionable insights suitable for executive decision-making.

## Output Requirements

#- Present the report using clear section headings.
#- Use bullet points wherever appropriate.
#- Generate a concise, consistent, and executive-ready business report.
#- Do not generate a report template.
#- Do not include placeholder fields such as Date, Prepared By, Company Name, or Contact Information.
#"""

#few_shot_user_prompt = PromptTemplate.from_template("""
#Analyze the following supply chain disruption dataset and prepare a comprehensive Business Impact Assessment Report by following the analytical style, report structure, and level of detail demonstrated in the reference examples.

## Reference Examples

#========================================================
### Example 1 – Port Congestion

#### Scenario

#A Tier-1 electronics supplier shipping semiconductor chips through the Port of Singapore experienced severe port congestion.

#### Observed Business Impact
#- Shipment delayed by 9 days.
#- Inventory dropped below the safety stock level.
#- Assembly line utilization decreased.
#- Logistics costs increased because of expedited air freight.

#### Analysis

#**Executive Summary**
#Port congestion disrupted inbound material flow, increasing logistics costs and creating moderate production risk.

#**Root Causes**
#- Port congestion
#- Vessel scheduling delays

#**Supplier Impact**
#The supplier fulfilled the order but experienced significant delivery delays.

#**Inventory Impact**
#Inventory levels declined below the required safety stock.

#**Production Risk**
#Production schedules required partial replanning.

#**Logistics Risk**
#Inbound transportation reliability decreased.

#**Financial Implications**
#Additional logistics expenses increased operational costs.

#**Recommended Mitigation Strategies**
#- Diversify shipping ports.
#- Increase safety stock for critical components.
#- Establish alternate logistics partners.

#**Business Continuity Assessment**
#Operations continued with manageable disruption.

#========================================================
### Example 2 – Supplier Failure

#### Scenario

#A Tier-2 supplier manufacturing industrial bearings declared bankruptcy, resulting in an immediate suspension of component deliveries.

#### Observed Business Impact
#- Production stopped for two manufacturing lines.
#- Inventory exhausted within five days.
#- Customer deliveries postponed.
#- Revenue at risk increased significantly.

#### Analysis

#**Executive Summary**
#Supplier failure created a critical supply interruption affecting production continuity.

#**Root Causes**
#- Supplier insolvency
#- Lack of alternate sourcing

#**Supplier Impact**
#Complete disruption of material supply.

#**Inventory Impact**
#Critical inventory depletion.

#**Production Risk**
#High production downtime.

#**Logistics Risk**
#Alternative sourcing required emergency transportation.

#**Financial Implications**
#Revenue losses and emergency procurement costs increased substantially.

#**Recommended Mitigation Strategies**
#- Qualify secondary suppliers.
#- Increase supplier financial monitoring.
#- Maintain strategic inventory reserves.

#**Business Continuity Assessment**
#Business continuity is at high risk without immediate intervention.

#========================================================
### Example 3 – Geopolitical Disruption

#### Scenario

#Trade restrictions between two countries disrupted cross-border transportation of electronic components used in multiple production facilities.

#### Observed Business Impact
#- Customs clearance delayed.
#- Lead time increased by 14 days.
#- Procurement costs increased.
#- Manufacturing schedules required revision.

#### Analysis

#**Executive Summary**
#Geopolitical instability significantly disrupted international material flow.

#**Root Causes**
#- Border restrictions
#- Trade sanctions
#- Customs inspections

#**Supplier Impact**
#International suppliers experienced prolonged delivery delays.

#**Inventory Impact**
#Material availability declined across multiple facilities.

#**Production Risk**
#Manufacturing schedules became uncertain.

#**Logistics Risk**
#International transportation routes were severely affected.

#**Financial Implications**
#Higher procurement and logistics costs reduced operating margins.

#**Recommended Mitigation Strategies**
#- Diversify supplier locations.
#- Increase regional sourcing.
#- Maintain alternate transportation routes.

#**Business Continuity Assessment**
#Operations continued under elevated supply chain risk.

#========================================================

## Input Data

#Supply Chain Disruption Dataset

#{business_context}

## Report Structure

#1. Executive Summary
#2. Identified Disruptions
#3. Root Causes
#4. Supplier Impact
#5. Inventory Impact
#6. Production Risk
#7. Logistics Risk
#8. Financial Implications
#9. Recommended Mitigation Strategies
#10. Business Continuity Assessment
#""")

#few_shot_prompt_text = few_shot_user_prompt.format(
    #business_context=business_context
#)

#response_few_shot = llm.invoke([
#    SystemMessage(content=few_shot_system_prompt),
#    HumanMessage(content=few_shot_prompt_text),
#])

#print("\n===== FEW-SHOT REPORT =====\n")
#print(response_few_shot.content)

#eval_zero_vs_few = llm.invoke(f"""
#You are evaluating two AI-generated Supply Chain Impact Assessment reports.
#
#---------- Report A: Zero-Shot ----------
#
#{response_zero_shot.content}
#
#---------- Report B: Few-Shot ----------
#
#{response_few_shot.content}
#
#Score both reports from 1 to 5 on:
#
#1. Completeness
#2. Business Reasoning
#3. Structure
#4. Actionability
#5. Executive Readiness
#
#Present the results as a markdown table with one row per criterion and one column per report.
#
#Then provide a short recommendation explaining the main strengths and weaknesses of each report.
#""")

#print("\n===== ZERO-SHOT vs FEW-SHOT EVALUATION =====\n")
#print(eval_zero_vs_few.content)

#cot_system_prompt = """
#You are a Senior Supply Chain Risk Analyst at TransGlobal Manufacturing.

#Expected Behavior

#- Act as an experienced supply chain risk analyst.
#- Analyze supply chain disruption data and generate evidence-based business impact assessment reports.
#- Base all observations, conclusions, and recommendations solely on the information provided.
#- Do not assume, infer, or fabricate facts that are not explicitly supported by the input.
#- If sufficient evidence is unavailable, clearly indicate that additional operational data is required.
#- Maintain a professional, objective, and business-oriented tone.
#- Produce concise, accurate, and actionable insights suitable for executive decision-making.
#- Provide only the final report unless the user explicitly requests your reasoning process.

#Output Requirements

#- Present the report using clear section headings.
#- Use bullet points wherever appropriate.
#- Generate a concise, logical, and executive-ready business report.
#- Do not generate a report template.
#- Do not include placeholder fields such as Date, Prepared By, Company Name, or Contact Information.
#"""

#cot_user_prompt = PromptTemplate.from_template("""
#Analyze the following supply chain disruption dataset and prepare a comprehensive Business Impact Assessment Report.

 #Input Data

#Supply Chain Disruption Dataset

#{business_context}

 #Task

#Perform a structured analysis of the provided dataset before generating the final report.

 #Report Structure

#1. Executive Summary
#2. Id- Use a structured reasoning approach to evaluate the available information before generating the final response.
#entified Disruptions
#3. Root Causes
#4. Supplier Impact
#5. Inventory Impact
#6. Production Risk
#7. Logistics Risk
#8. Financial Implications
#9. Recommended Mitigation Strategies
#10. Business Continuity Assessment
#""")

#cot_prompt_text = cot_user_prompt.format(
    #business_context=business_context
#)

#response_cot = llm.invoke([
    #SystemMessage(content=cot_system_prompt),
    #HumanMessage(content=cot_prompt_text),
#])

#print("\n===== CHAIN-OF-THOUGHT REPORT =====\n")
#print(response_cot.content)

#tot_system_prompt = """
#You are a Senior Supply Chain Risk Analyst at TransGlobal Manufacturing.

#Expected Behavior

#- Act as an experienced supply chain risk analyst.
#- Analyze supply chain disruption data and generate evidence-based Business Impact Assessment Reports.
#- Consider multiple plausible solution approaches before selecting the most appropriate recommendation.
#- Evaluate alternative mitigation strategies based on business impact, feasibility, implementation effort, cost, and business continuity.
#- Compare the strengths and trade-offs of each alternative before arriving at the final recommendation.
#- Base all observations, conclusions, and recommendations solely on the information provided.
#- Do not assume, infer, or fabricate facts that are not explicitly supported by the input.
##- If sufficient evidence is unavailable, clearly indicate that additional operational data is required.
#- Maintain a professional, objective, and business-oriented tone.
#- Produce concise, accurate, and actionable insights suitable for executive decision-making.
#- Provide only the final report unless the user explicitly requests your reasoning process.

#Output Requirements

#- Present the report using clear section headings.
#- Use bullet points wherever appropriate.
#- Generate a concise, logical, and executive-ready business report.
#- Do not generate a report template.
#- Do not include placeholder fields such as Date, Prepared By, Company Name, or Contact Information.
#"""

#tot_user_prompt = PromptTemplate.from_template("""
#Analyze the following supply chain disruption dataset and prepare a comprehensive Business Impact Assessment Report.

#Input Data

#Supply Chain Disruption Dataset

#{business_context}

#Task

#Develop mitigation recommendations by considering multiple feasible approaches for each major disruption and recommend the most appropriate strategy based on the available evidence.

#Report Structure

#1. Executive Summary
#2. Identified Disruptions
#3. Root Causes
#4. Supplier Impact
#5. Inventory Impact
#6. Production Risk
#7. Logistics Risk
#8. Financial Implications
#9. Recommended Mitigation Strategies
#10. Business Continuity Assessment
#""")

#tot_prompt_text = tot_user_prompt.format(
    #business_context=business_context
#)

#response_tot = llm.invoke([
    #SystemMessage(content=tot_system_prompt),
    #HumanMessage(content=tot_prompt_text),
#])

#print("\n===== TREE-OF-THOUGHT REPORT =====\n")
#print(response_tot.content)

#eval_cot_vs_tot = llm.invoke(f"""
#You are evaluating two AI-generated Supply Chain Impact Assessment reports.

#---------- Report A: Chain-of-Thought ----------

#{response_cot.content}

#---------- Report B: Tree-of-Thought ----------

#{response_tot.content}

#Score both reports from 1 to 5 on:

#1. Completeness
#2. Business Reasoning
#3. Structure
#4. Actionability
#5. Executive Readiness

#Present the results as a markdown table with one row per criterion and one column per report.

#Then provide a short recommendation explaining the main strengths and weaknesses of each report.
#""")

#print("\n===== CoT vs ToT EVALUATION =====\n")
#print(eval_cot_vs_tot.content)

tot_system_prompt = """
You are a Senior Supply Chain Risk Analyst at TransGlobal Manufacturing.

Expected Behavior

- Act as an experienced supply chain risk analyst.
- Analyze supply chain disruption data and generate evidence-based Business Impact Assessment Reports.
- Consider multiple plausible solution approaches before selecting the most appropriate recommendation.
- Evaluate alternative mitigation strategies based on business impact, feasibility, implementation effort, cost, and business continuity.
- Compare the strengths and trade-offs of each alternative before arriving at the final recommendation.
- Base all observations, conclusions, and recommendations solely on the information provided.
- Do not assume, infer, or fabricate facts that are not explicitly supported by the input.
- If sufficient evidence is unavailable, clearly indicate that additional operational data is required.
- Maintain a professional, objective, and business-oriented tone.
- Produce concise, accurate, and actionable insights suitable for executive decision-making.
- Provide only the final report unless the user explicitly requests your reasoning process.

Output Requirements

- Present the report using clear section headings.
- Use bullet points wherever appropriate.
- Generate a concise, logical, and executive-ready business report.
- Do not generate a report template.
- Do not include placeholder fields such as Date, Prepared By, Company Name, or Contact Information.
"""

tot_user_prompt = PromptTemplate.from_template("""
Analyze the following supply chain disruption dataset and prepare a comprehensive Business Impact Assessment Report.

Input Data

Supply Chain Disruption Dataset

{business_context}

Task

Develop mitigation recommendations by considering multiple feasible approaches for each major disruption and recommend the most appropriate strategy based on the available evidence.

Report Structure

1. Executive Summary
2. Identified Disruptions
3. Root Causes
4. Supplier Impact
5. Inventory Impact
6. Production Risk
7. Logistics Risk
8. Financial Implications
9. Recommended Mitigation Strategies
10. Business Continuity Assessment
""")

def run_with_params(temperature, max_tokens, top_p):
    tuned_llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    messages = [
        SystemMessage(content=tot_system_prompt),
        HumanMessage(
            content=tot_user_prompt.format(
                business_context=business_context
            )
        ),
    ]

    return tuned_llm.invoke(messages).content

#temperature_results = {}

#for temp in [0.0, 0.7, 1.2]:
    #report = run_with_params(
        #temperature=temp,
        #max_tokens=500,
        #top_p=1.0
    #)

    #temperature_results[temp] = report

    #print(f"\n===== TEMPERATURE = {temp} =====\n")
    #print(report)

    #print("\nWord Count:", len(report.split()))

#token_results = {}

#for tokens in [150, 400, 1200]:
    #report = run_with_params(
        #temperature=0.0,
        #max_tokens=tokens,
        #top_p=1.0
    #)

    #token_results[tokens] = report

    #print(f"\n===== MAX TOKENS = {tokens} =====\n")
    #print(report)

    #print("\nWord Count:", len(report.split()))

#topp_results = {}

#for topp in [0.1, 0.5, 1.0]:
    #report = run_with_params(
        #temperature=0.8,
        #max_tokens=700,
        #top_p=topp
    #)

    #topp_results[topp] = report

    #print(f"\n===== TOP_P = {topp} =====\n")
    #print(report)

    #print("\nWord Count:", len(report.split()))

# CEO: concise and consistent
ceo_report = run_with_params(
    temperature=0.2,
    max_tokens=450,
    top_p=0.5
)

# Operations Manager: balanced detail
operations_report = run_with_params(
    temperature=0.5,
    max_tokens=900,
    top_p=0.7
)

# Risk Officer: comprehensive analytical narrative
risk_report = run_with_params(
    temperature=0.8,
    max_tokens=1500,
    top_p=0.9
)

print("\n===== CEO REPORT =====\n")
print(ceo_report)

print("\n===== OPERATIONS MANAGER REPORT =====\n")
print(operations_report)

print("\n===== RISK OFFICER REPORT =====\n")
print(risk_report)

print("\n===== WORD COUNT SUMMARY =====")
print("CEO:", len(ceo_report.split()))
print("Operations Manager:", len(operations_report.split()))
print("Risk Officer:", len(risk_report.split()))