def generate_scorecard(tagged_transcript, client):
    prompt = f"""
    You are a Quality Assurance Auditor. Evaluate the following emotion-tagged transcript based on the exact rubric below.
    
    STRICT SCORING RULES - READ CAREFULLY:
    1. DEFAULT TO MAX SCORE: You MUST award the MAXIMUM possible score for every single parameter by default. 
    2. BURDEN OF PROOF & TRANSCRIPT ERRORS: AI transcripts often cut off mid-sentence or miss words. You are strictly FORBIDDEN from deducting marks for incomplete sentences or assuming the agent failed just because the transcript is fragmented. If you deduct marks, your justification MUST contain a direct quote proving a clear agent failure. If no absolute negative proof exists, you MUST award the max score.
    3. EXACT MARKS ONLY: You are ONLY permitted to use the exact numeric values listed below.
    4. GRAMMAR & PUNCTUATION (Param 7): The transcription AI naturally makes punctuation and grammar mistakes. Do NOT deduct marks for missing punctuation or minor grammatical errors. Only deduct if there is undeniable proof of a massive, unprofessional linguistic failure by the agent.

    PARAMETER SPECIFIC DEFINITIONS:
    - Opening Protocol (Param 1): The agent must completely finish their opening greeting within the first 10 seconds of the call (by 00:10). If the opening spills past 10 seconds, deduct marks.
    - Took Ownership (Param 3): The agent proactively checks information without forcing the customer to repeat details they already provided. Note: Asking a question to *confirm* details is good practice and MUST NOT be penalized.
    - Clarity / Fumbling (Param 6): Fumbling is strictly defined as the agent using repeated filler words. Only deduct marks if there is clear evidence of the agent relying on repeated filler words.
    - Empathy and Sympathy (Param 8): Do not deduct marks just because the agent failed to use specific empathetic phrases. Only deduct marks if the agent actively misbehaves, is dismissive, or is rude to the customer.
    - Probing (Param 10): Asking specific, necessary questions. Only deduct marks if probing was clearly required to resolve the issue and the agent completely failed to ask.
    - Interruption & Assurance (Param 12): Assurance means giving the customer a timeline or guarantee of when something will happen. Deduct marks if the situation called for a timeline or guarantee and the agent failed to provide one.
    - Closing Protocol (Param 15): The agent MUST ask if there is anything else they can assist the customer with, AND they MUST say "thank you".

    RUBRIC & ALLOWED SCORES:
    1. Opening Protocol followed (Allowed Marks: 0 or 3)
    2. Followed Customer's Language (Allowed Marks: 0 or 2)
    3. Took Ownership/ Went extra Miles (Allowed Marks: 0, 5, or 10)
    4. Was the responses aligned with cx concern? (Allowed Marks: 0, 3, or 5)
    5. Was polite and Courteous throughout the Call (Allowed Marks: 0, 5, or 10)
    6. Had CLARITY of voice/fumbling/rate of speech (Allowed Marks: 0, 3, or 6)
    7. Grammatical/Pronunciation Error (Allowed Marks: 0, 3, or 5)
    8. Empathy and Sympathy (Allowed Marks: 0, 4, or 8)
    9. Avoided Dead Air (Allowed Marks: 0 or 3)
    10. Was probing done on the call/Proactivness (Allowed Marks: 0, 4, or 8)
    11. Followed hold & mute protocol (Allowed Marks: 0, 5, or 10)
    12. Interruption & Assurance (Allowed Marks: 0, 5, or 10)
    13. Effective listening (Allowed Marks: 0, 3, or 5)
    14. Process knowledge/Tat (Allowed Marks: 0, 5, or 10)
    15. Followed closing protocol (Allowed Marks: 0, 3, or 5)

    Output a strict Markdown table with columns: | Parameter | Score Awarded | Max Score | Justification (Must include exact quote if marks are deducted) |
    At the very bottom, bold the Final Total Score out of 100. Do not include any other text.

    TRANSCRIPT:
    {tagged_transcript}
    """
    return call_groq_text_model(prompt, client, required_char="|")
