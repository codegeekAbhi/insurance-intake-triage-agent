import json
import re
import time
import pandas as pd

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT_V1 = """You are an insurance intake triage assistant for Harper, a commercial insurance company.

Given a free text description of a business, classify it and respond with JSON only, no other text.

Output schema:
{
  "subtype": one of "used_car_dealer", "tow_operator", "freight_carrier", "auto_repair_shop",
  "coverage_lines": a list using only these labels: "General Liability", "Commercial Auto", "Garage Liability", "Inland Marine", "Umbrella", "Workers Compensation",
  "urgency": one of "low", "medium", "high",
  "routing": one of "web", "voice", "human",
  "confidence": a number from 0 to 1 reflecting your certainty,
  "rationale": a one sentence reason
}

Guidance:
- Garage Liability applies to businesses that service, store, or move vehicles for others (tow operators, repair shops, dealers).
- Inland Marine applies when cargo or goods in transit need coverage (freight carriers).
- Urgency is high when there is an explicit or implicit deadline (expiring plates, a contract start date, a lapsing policy).
- Routing should be human whenever urgency is high or the situation is unusual, otherwise web is fine for routine cases.
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1 + """

Examples of edge cases to handle carefully:

Example 1:
Business: "Freight brokerage that books loads for other carriers but does not own trucks."
Correct output: subtype is freight_carrier, coverage_lines is only ["General Liability"], because a brokerage with no owned trucks does not need Commercial Auto or Inland Marine.

Example 2:
Business: "Auto repair shop owner asking about home based garage coverage, business not yet operating."
Correct output: subtype is auto_repair_shop, coverage_lines is only ["General Liability"], because Garage Liability applies once the business is actively servicing vehicles for customers, not before.

Use the same reasoning pattern for similar edge cases: a business that does not yet own or operate the physical assets a coverage line protects does not need that line yet.
"""

EVAL_SET = [
    {"description": "Family owned used car dealership in Ohio with about 40 vehicles on the lot. Dealer plates renew next week and they want to make sure coverage does not lapse.",
     "subtype": "used_car_dealer", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "high", "routing": "human"},
    {"description": "Small tow truck operator in Texas running two trucks, mostly highway breakdowns and local impound work. Looking for standard liability coverage, nothing urgent.",
     "subtype": "tow_operator", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "low", "routing": "web"},
    {"description": "Regional freight carrier hauling refrigerated produce across five states with a 25 truck fleet. A new contract with a grocery chain starts in 10 days and they need cargo coverage in place before then.",
     "subtype": "freight_carrier", "coverage_lines": ["Commercial Auto", "Inland Marine", "General Liability"], "urgency": "high", "routing": "human"},
    {"description": "Independent auto repair shop in Arizona, three bays, mostly oil changes and brake work, current policy renews in three months.",
     "subtype": "auto_repair_shop", "coverage_lines": ["Garage Liability", "General Liability"], "urgency": "low", "routing": "web"},
    {"description": "Used car dealer in Florida expanding from 20 to 60 vehicles, no immediate deadline but wants to make sure their auto coverage scales with the new inventory.",
     "subtype": "used_car_dealer", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "medium", "routing": "web"},
    {"description": "Single owner tow operator whose current insurer just dropped them, with a policy lapsing in four days, needs new coverage immediately to keep operating.",
     "subtype": "tow_operator", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "high", "routing": "human"},
    {"description": "Freight brokerage that books loads for other carriers but does not own trucks, asking what coverage they actually need.",
     "subtype": "freight_carrier", "coverage_lines": ["General Liability"], "urgency": "low", "routing": "voice"},
    {"description": "Auto repair shop with two locations in Nevada, recently had a customer vehicle damaged in their lot and wants better liability protection going forward.",
     "subtype": "auto_repair_shop", "coverage_lines": ["Garage Liability", "General Liability"], "urgency": "medium", "routing": "voice"},
    {"description": "Used car dealer who also tows trade in vehicles to auction, six person staff, wants one quote covering both the lot and the tow truck they use.",
     "subtype": "used_car_dealer", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "low", "routing": "web"},
    {"description": "Long haul trucking company with 50 trucks moving electronics. Their current cargo policy excludes electronics and they found out after a claim was denied last month.",
     "subtype": "freight_carrier", "coverage_lines": ["Commercial Auto", "Inland Marine", "General Liability"], "urgency": "high", "routing": "human"},
    {"description": "Small tow company just starting out, one truck, owner wants to understand what coverage is even required before they start operating next month.",
     "subtype": "tow_operator", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "medium", "routing": "voice"},
    {"description": "Auto repair shop owner asking general questions about whether their homeowners policy could cover their home based garage, business not yet operating.",
     "subtype": "auto_repair_shop", "coverage_lines": ["General Liability"], "urgency": "low", "routing": "voice"},
    {"description": "Used car dealership chain with locations in three states, looking to consolidate four separate state policies into one before their fiscal year end in two weeks.",
     "subtype": "used_car_dealer", "coverage_lines": ["Garage Liability", "Commercial Auto", "Umbrella"], "urgency": "high", "routing": "human"},
    {"description": "Freight carrier moving construction equipment, fleet of 10 flatbed trucks, current policy is fine, just shopping around for a better rate.",
     "subtype": "freight_carrier", "coverage_lines": ["Commercial Auto", "Inland Marine"], "urgency": "low", "routing": "web"},
    {"description": "Tow operator whose insurer is requiring a workers compensation policy before renewal because they just hired their first two employees.",
     "subtype": "tow_operator", "coverage_lines": ["Garage Liability", "Commercial Auto", "Workers Compensation"], "urgency": "high", "routing": "human"},
    {"description": "Auto repair shop wants to add a second bay and a tow truck to pick up customer vehicles, asking what additional coverage that would require.",
     "subtype": "auto_repair_shop", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "medium", "routing": "voice"},
    {"description": "Used car dealer with a routine annual renewal coming up in two months, no changes to their operation, just wants to confirm pricing.",
     "subtype": "used_car_dealer", "coverage_lines": ["Garage Liability", "Commercial Auto"], "urgency": "low", "routing": "web"},
    {"description": "Freight carrier that had a truck fire destroy a load of pharmaceuticals last week and needs to file a claim and understand their cargo coverage limits immediately.",
     "subtype": "freight_carrier", "coverage_lines": ["Commercial Auto", "Inland Marine"], "urgency": "high", "routing": "human"},
]


def parse_json_response(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def triage(client, description, system_prompt=SYSTEM_PROMPT_V1, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return parse_json_response(raw)
    except Exception as e:
        return {"error": str(e), "raw": raw}


def triage_with_guardrail(client, description, system_prompt=SYSTEM_PROMPT_V1, threshold=0.7):
    predicted = triage(client, description, system_prompt=system_prompt)
    if "error" not in predicted and predicted.get("confidence", 1.0) <= threshold:
        predicted["routing"] = "human"
        predicted["guardrail_triggered"] = True
    else:
        predicted["guardrail_triggered"] = False
    return predicted


def score_case(predicted, expected):
    if "error" in predicted:
        return {"subtype_correct": False, "coverage_overlap": 0.0, "urgency_correct": False, "routing_correct": False}
    pred_coverage = set(predicted.get("coverage_lines", []))
    exp_coverage = set(expected["coverage_lines"])
    overlap = len(pred_coverage & exp_coverage) / len(exp_coverage) if exp_coverage else 0.0
    return {
        "subtype_correct": predicted.get("subtype") == expected["subtype"],
        "coverage_overlap": overlap,
        "urgency_correct": predicted.get("urgency") == expected["urgency"],
        "routing_correct": predicted.get("routing") == expected["routing"],
    }


def run_eval(client, eval_set, system_prompt=SYSTEM_PROMPT_V1, label="run", progress_callback=None):
    rows = []
    total = len(eval_set)
    for i, case in enumerate(eval_set):
        predicted = triage(client, case["description"], system_prompt=system_prompt)
        scores = score_case(predicted, case)
        rows.append({"description": case["description"][:60] + "...", **scores})
        if progress_callback:
            progress_callback((i + 1) / total)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    summary = {
        "label": label,
        "subtype_accuracy": df["subtype_correct"].mean(),
        "coverage_overlap": df["coverage_overlap"].mean(),
        "urgency_accuracy": df["urgency_correct"].mean(),
        "routing_accuracy": df["routing_correct"].mean(),
    }
    return df, summary
