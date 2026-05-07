"""
LLM Narrator — Traffic Safety Explanations
src/modeling/narrator.py

Encapsulates the logic for calling the Gemini API to turn SHAP feature 
impacts into human-readable narratives.
"""

import os
import google.generativeai as genai

def generate_explanation(segment_id, risk_score, top_factors, model_name="gemini-2.5-flash", stream=False):
    """
    Sends SHAP feature impacts to Gemini to generate a human-readable explanation.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[!] GEMINI_API_KEY environment variable not found. Skipping narrative.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # Format the factors into a prompt
        factors_str = "\n".join([f"- {feat}: {val:.4f} impact" for feat, val in top_factors.items()])

        prompt = f"""
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

        if stream:
            try:
                response = model.generate_content(prompt, stream=True)
                def chunk_generator():
                    try:
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                    except Exception as gen_e:
                        yield f"\n\n⚠️ [Streaming Error] {gen_e}"
                return chunk_generator()
            except Exception as e:
                err = f"Error starting stream: {e}"
                print(err)
                def error_gen(): yield f"⚠️ {err}"
                return error_gen()
        else:
            response = model.generate_content(prompt)
            return response.text.strip()
        
    except Exception as e:
        error_msg = f"Error in LLM Narrative generation: {e}"
        print(error_msg)
        # Return a generator that yields the error if streaming was requested
        if stream:
            def error_gen(): yield f"⚠️ {error_msg}"
            return error_gen()
        return None
