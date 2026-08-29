"""
prompts.py
All prompt engineering lives here: the safety system prompt, the JSON
schema instructions, a PromptTemplate (single-string) version, and a
ChatPromptTemplate (System + Human) version used by the chains.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ---------------------------------------------------------------------------
# SAFETY SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_SAFETY_PROMPT = """You are MediGuide AI, an EDUCATIONAL medical information assistant.

Non-negotiable safety rules you must always follow:
1. You are NOT a doctor and must never present a confirmed diagnosis.
2. Always frame information as general, educational guidance only.
3. Always encourage the user to consult a licensed healthcare professional.
4. If symptoms suggest a potentially life-threatening situation (e.g. severe
   chest pain, difficulty breathing, signs of stroke, severe bleeding,
   suicidal ideation), classify urgency as "EMERGENCY" and clearly instruct
   the user to seek emergency care immediately (call local emergency
   services or go to the nearest emergency room).
5. Never recommend specific prescription medications or dosages.
6. Be calm, clear, and reassuring in tone, never alarmist, but never
   downplay genuinely serious symptoms.
7. Respond in the language requested by the user.
8. When asked for JSON, your entire response must be a SINGLE valid JSON
   object and nothing else -- no markdown fences, no preamble, no
   explanation outside the JSON.
"""

# ---------------------------------------------------------------------------
# JSON SCHEMA INSTRUCTIONS
# ---------------------------------------------------------------------------
JSON_SCHEMA_INSTRUCTIONS = """Return ONLY a valid JSON object with EXACTLY this structure
(no extra keys, no missing keys, no ```json fences, no commentary):

{{
  "summary": "one paragraph plain-language summary of the patient's situation",
  "possible_conditions": [
    {{"name": "condition name", "reason": "why it's plausible, in plain language"}}
  ],
  "urgency_level": "LOW | MEDIUM | HIGH | EMERGENCY",
  "recommended_next_steps": ["step 1", "step 2", "..."],
  "questions_for_doctor": ["question 1", "question 2", "..."],
  "warning_signs": ["sign 1", "sign 2", "..."]
}}
"""

_PATIENT_BLOCK = (
    "Patient information:\n"
    "- Age: {age}\n"
    "- Gender: {gender}\n"
    "- Symptoms: {symptoms}\n"
    "- Duration: {duration}\n"
    "- Severity (1-10): {severity}\n"
    "- Existing conditions: {existing_conditions}\n"
    "- Current medications: {medications}\n"
    "- Additional notes: {notes}\n\n"
    "Respond in this language: {language}.\n\n"
)

# ---------------------------------------------------------------------------
# PromptTemplate -- single-string reusable template (required by assignment)
# ---------------------------------------------------------------------------
ASSESSMENT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "age", "gender", "symptoms", "duration", "severity",
        "existing_conditions", "medications", "notes", "language",
    ],
    template=SYSTEM_SAFETY_PROMPT + "\n\n" + _PATIENT_BLOCK + JSON_SCHEMA_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# ChatPromptTemplate -- System + Human conversation (required by assignment)
# ---------------------------------------------------------------------------
ASSESSMENT_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_SAFETY_PROMPT),
    ("human", _PATIENT_BLOCK + JSON_SCHEMA_INSTRUCTIONS),
])

# ---------------------------------------------------------------------------
# Narrative streaming template -- human-readable version for st.write_stream
# ---------------------------------------------------------------------------
NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_SAFETY_PROMPT + "\n\nFor this response, do NOT return JSON. "
               "Instead, write a short, warm, plain-language narrative (4-6 sentences) "
               "summarizing the patient's situation, the urgency level, and the single "
               "most important next step. End with a reminder to consult a healthcare "
               "professional."),
    ("human", _PATIENT_BLOCK.replace("\n\nRespond", "\nRespond")),
])
