import ollama
import json
import os

BASE = r"C:\Users\RileyMoorman\Documents\Week 2 GenAI Curriculum\agentic_ai"
MODEL = "llama3.2:3b"


# ── Tool functions ────────────────────────────────────────────────────────────

def calculate_bmi(weight, height) -> dict:
    weight, height = float(weight), float(height)
    if height <= 0 or weight <= 0:
        return {"error": "Weight and height must be positive values."}

    bmi = weight / (height ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "bmi": round(bmi, 2),
        "category": category,
        "weight_kg": weight,
        "height_m": height,
    }


def lookup_drug_interaction(drug_a: str, drug_b: str) -> dict:
    drug_a = drug_a.strip().lower()
    drug_b = drug_b.strip().lower()

    keys_to_try = [
        f"{drug_a}_{drug_b}",
        f"{drug_b}_{drug_a}",
    ]

    for filename in os.listdir(BASE):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(BASE, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in keys_to_try:
            if key in data:
                return {"drug_a": drug_a, "drug_b": drug_b, "interaction": data[key]}

        if isinstance(data, list):
            for record in data:
                drugs = {record.get("drug_a", "").lower(), record.get("drug_b", "").lower()}
                if drugs == {drug_a, drug_b}:
                    return record

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "interaction": "No known interaction found in local database.",
    }


# ── Tool definitions (Ollama format) ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": (
                "Calculate BMI given weight in kilograms and height in meters. "
                "Convert imperial units before calling: divide lbs by 2.205 for kg, "
                "multiply total inches by 0.0254 for meters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight": {"type": "number", "description": "Weight in kilograms"},
                    "height": {"type": "number", "description": "Height in meters"},
                },
                "required": ["weight", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_interaction",
            "description": "Look up the interaction between two drugs. Returns known risks or a no-interaction-found message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_a": {"type": "string", "description": "First drug name"},
                    "drug_b": {"type": "string", "description": "Second drug name"},
                },
                "required": ["drug_a", "drug_b"],
            },
        },
    },
]

TOOL_MAP = {
    "calculate_bmi": calculate_bmi,
    "lookup_drug_interaction": lookup_drug_interaction,
}


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(prompt: str):
    messages = [{"role": "user", "content": prompt}]

    print(f"\nPrompt: {prompt}\n")
    print("=" * 60)

    while True:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        messages.append(response.message)

        if not response.message.tool_calls:
            print(f"\nAgent Summary:\n{response.message.content}")
            break

        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"\n[Calling] {name}({args})")

            result = TOOL_MAP[name](**args)
            print(f"[Result]  {result}")

            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prompt = (
        "Patient takes metformin, weighs 146 lbs at 5'4\", and is complaining of "
        "joint pain — can we prescribe ibuprofen? Calculate their BMI and check interactions."
    )
    run_agent(prompt)
