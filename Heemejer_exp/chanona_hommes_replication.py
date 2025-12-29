#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 10:48:03 2025
"""

import math
import json
import numpy as np
from openai import OpenAI

client = OpenAI()

N_AGENTS = 2
T = 5

def price_update(market, avg_forecast, eps):
    if market == "positive":
        return (20/21) * (avg_forecast + 3) + eps
    elif market == "negative":
        return (20/21) * (123 - avg_forecast) + eps
    else:
        raise ValueError("market must be positive or negative")

def earnings(p, v):
    # Eq (3): max(1300 - (1300/49)*(p-v)^2, 0)
    return max(1300 - (1300/49) * (p - v) ** 2, 0)

# For Responses API: Structured Outputs vs JSON mode
TEXT_FORMAT_SCHEMA = {
    "format": {
        "type": "json_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "predictedValue": {"type": "number"},
            },
            "required": ["reasoning", "predictedValue"],
            "additionalProperties": False,
        },
    }
}

TEXT_FORMAT_JSON = {
    "format": {"type": "json_object"}
}

# For Chat Completions:
# Use JSON mode (guarantees valid JSON). We'll validate keys ourselves.
RESPONSE_FORMAT = {"type": "json_object"}

def call_agent(model, system_blocks, history_assistant_msgs, market_info_user_msg,
               temperature=0.7, seed=None):
    messages = [{"role": "system", "content": b} for b in system_blocks]
    messages += history_assistant_msgs
    messages.append({"role": "user", "content": market_info_user_msg})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
        temperature=temperature,
        seed=seed,
    )

    txt = resp.choices[0].message.content
    obj = json.loads(txt)

    # Minimal validation (helps catch weird outputs early)
    if "predictedValue" not in obj:
        raise ValueError(f"Missing predictedValue in response: {obj}")
    if "reasoning" not in obj:
        raise ValueError(f"Missing reasoning in response: {obj}")

    return obj



def run_one_market(model, market="positive", memory_m=3, temperature=1.0,
                   seed_base=1234, reasoning_effort=None):
    rng = np.random.default_rng(seed_base)
    # per paper: eps ~ N(0, 1/4) => std=0.5
    eps_std = 0.5

    # per-agent state
    prices = []                     # realized P(1..t-1)
    preds = [[] for _ in range(N_AGENTS)]      # each agent's V history
    earnings_tot = [0.0] * N_AGENTS
    assistant_hist = [[] for _ in range(N_AGENTS)]  # store last m JSON outputs as assistant msgs

    # system prompt blocks: you’d paste/adapt Appendix B’s system messages here.
    system_blocks = [
    "You are a forecasting agent in a repeated market experiment. "
    "You must forecast next period's price to maximize earnings based on accuracy.",
    "Return JSON only with keys reasoning (30-50 words) and predictedValue (number)."
]


    for t in range(1, T + 1):
        # 1) collect forecasts
        forecasts = []
        for i in range(N_AGENTS):
            # memory: keep only last m assistant messages
            hist_i = assistant_hist[i][-memory_m:] if memory_m > 0 else []

            market_info = {
                "t": t,
                "market_prices": list(reversed(prices)),            # [P(t-1),...,P(1)]
                "your_predictions": list(reversed(preds[i])),      # [V(t-1),...,V(1)]
                "total_earnings": earnings_tot[i],
            }
            user_msg = f"Here is the data:\n{market_info}\nReturn JSON only."

            out = call_agent(
                model=model,
                system_blocks=system_blocks,
                history_assistant_msgs=hist_i,
                market_info_user_msg=user_msg,
                temperature=temperature,
                seed=seed_base + 10000*t + i,
            )

            v = float(out["predictedValue"])
            v = max(0.0, min(v, 200.0))  # debug clamp; remove later if you want
            forecasts.append(v)
            preds[i].append(v)

            # store agent output as an assistant message for memory
            assistant_hist[i].append({"role": "assistant", "content": json.dumps(out)})

        # 2) price update
        avg = sum(forecasts) / N_AGENTS
        eps = rng.normal(0.0, eps_std)
        p = price_update(market, avg, eps)

        # enforce period-1 bounds from instructions if you want to be strict
        if t == 1:
            p = min(max(p, 0.0), 100.0)
        else:
            p = max(p, 0.0)

        prices.append(p)

        # 3) earnings
        for i in range(N_AGENTS):
            e = earnings(p, preds[i][-1])
            earnings_tot[i] += e

    return prices, preds, earnings_tot

if __name__ == "__main__":
    prices, preds, earns = run_one_market(
        model="gpt-5-chat-latest",
        market="positive",
        memory_m=1,
        temperature=0.7,
    )
    print("prices:", prices)
    print("preds:", preds)
    print("earnings:", earns)
