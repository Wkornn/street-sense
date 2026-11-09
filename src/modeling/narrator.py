"""
LLM Narrator — Traffic Safety Explanations
src/modeling/narrator.py

Encapsulates the logic for calling the Gemini API to turn SHAP feature 
impacts into human-readable narratives.
"""

import os

from google import genai


def _prompt(segment_id, risk_score, top_factors):
    factors_str = "\n".join(f"- {feat}: {val:.4f} impact" for feat, val in top_factors.items())
    return f"""
        You are an expert urban traffic safety analyst in Bangkok. I have an AI model that predicts
        accident risk for road segments.

        Road Segment ID: {segment_id}
        Predicted Risk Probability: {risk_score * 100:.1f}%

        Our SHAP analysis identified these top factors driving this specific prediction
        (positive = increases risk, negative = decreases risk):
        {factors_str}

        Task:
        1. Write a 3-sentence professional explanation for a city official explaining WHY
           this specific road is dangerous based on these factors.
        2. Suggest one specific, actionable engineering or enforcement countermeasure.

        Style: Concise, data-driven, and local to Bangkok context. Do not use markdown.
        """


def generate_explanation(segment_id, risk_score, top_factors, model_name="gemini-2.5-flash", stream=False):
    """Ask Gemini to explain a segment's risk from its top SHAP factors.

    Returns the text, or (stream=True) a generator of text chunks. Returns None if
    GEMINI_API_KEY is not set. Errors are returned as text, never raised, so the app keeps working.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[!] GEMINI_API_KEY environment variable not found. Skipping narrative.")
        return None

    prompt = _prompt(segment_id, risk_score, top_factors)
    try:
        client = genai.Client(api_key=api_key)
        if not stream:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()

        # The stream is started inside the generator, so the generator keeps `client` alive while
        # the app reads it (the client closes its connection once nothing refers to it).
        def chunk_generator():
            try:
                for chunk in client.models.generate_content_stream(model=model_name, contents=prompt):
                    if chunk.text:
                        yield chunk.text
            except Exception as e:  # noqa: BLE001 - LLM errors are shown in the text, not raised
                yield f"\n\n⚠️ [Streaming Error] {e}"

        return chunk_generator()

    except Exception as e:  # noqa: BLE001 - Gemini raises many error types; never crash the app
        error_msg = f"Error in LLM Narrative generation: {e}"
        print(error_msg)
        if stream:
            def error_gen():
                yield f"⚠️ {error_msg}"
            return error_gen()
        return None
