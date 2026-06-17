# insurance-intake-triage-agent
AI agent that triages commercial insurance intake by business type, urgency, and routing, with a labeled eval harness and a confidence based human handoff guardrail. Built for Harper's automotive and transportation vertical, using Groq and Llama 3.3 70B

# Harper Insurance Triage Agent

An AI agent that triages commercial insurance intake by business subtype, coverage need, urgency, and routing, with a labeled eval harness and a confidence based human handoff guardrail. Built for Harper's automotive and transportation vertical.

Link : https://harper-triage-agent.streamlit.app/

## What it does

1. Takes a free text business description as input
2. Classifies the business subtype: used car dealer, tow operator, freight carrier, or auto repair shop
3. Recommends coverage lines from Harper's actual product list (General Liability, Commercial Auto, Garage Liability, Inland Marine, Umbrella, Workers Compensation)
4. Scores urgency based on explicit or implicit deadlines
5. Decides routing: web, voice, or human
6. Applies a confidence threshold that forces human handoff when the model is not sure, rather than guessing

## Why this exists

Most agent demos stop at "it works." This one is built around an eval harness instead. There is a labeled golden set of 18 realistic scenarios, a baseline accuracy run, one round of prompt iteration based on the model's actual failure cases, and the resulting accuracy delta. The agent itself is the small part. The eval harness, and the discipline of measuring before changing anything, is the actual point.

## Stack

- Groq API, Llama 3.3 70B (free tier)
- Python and pandas
- No paid services required

## Setup

1. Get a free Groq API key at console.groq.com
2. Clone this repo
3. Run `pip install groq pandas`
4. Run `python harper_triage_agent.py`, it will prompt for your API key at runtime, the key is never hardcoded or stored in this repo

## Eval methodology

Each scenario in the golden set has a hand labeled correct subtype, coverage lines, urgency, and routing. The agent is scored on exact match for subtype, urgency, and routing, and on overlap for coverage lines since a partially correct recommendation is a smaller miss than the wrong industry entirely. The script runs a baseline, surfaces the failure cases, adds two few shot examples targeting those exact failures, and reruns the eval to show the improvement.

## Results

See the printed comparison table at the end of the script for baseline versus post iteration accuracy across all four metrics, plus the guardrail trigger rate showing how often low confidence cases were correctly routed to a human.

## Extending

The same pattern, classify, recommend coverage, score urgency, route with a guardrail, generalizes to any of Harper's other verticals: care providers, hospitality, professional services. Swap in a new golden set and system prompt, the eval harness stays the same.
