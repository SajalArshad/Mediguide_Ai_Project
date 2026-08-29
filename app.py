"""
app.py
MediGuide AI -- Streamlit UI.
Run with: streamlit run app.py

Educational prototype only. Not a medical device, not a substitute for a
licensed healthcare professional.
"""

import streamlit as st

from src.config import (
    APP_NAME, APP_TAGLINE, AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
    GENDER_OPTIONS, SYMPTOM_OPTIONS, DURATION_OPTIONS, LANGUAGE_OPTIONS,
    CACHE_OPTIONS, resolve_api_key,
)
from src.cache_manager import apply_cache_choice
from src.chains import build_llm, build_assessment_chain, run_assessment_chain, stream_narrative
from src.utils import safe_parse_json, format_symptom_summary, urgency_badge

st.set_page_config(page_title=APP_NAME, page_icon="\U0001FA7A", layout="wide")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"\U0001FA7A {APP_NAME}")
    st.caption(APP_TAGLINE)

    st.warning(
        "**Medical disclaimer:** MediGuide AI is an educational prototype. "
        "It is NOT a doctor and does NOT provide a diagnosis. Always consult "
        "a licensed healthcare professional. In an emergency, call your local "
        "emergency number immediately.",
        icon="\u26A0\uFE0F",
    )

    st.subheader("\U0001F511 Your OpenAI API key")
    user_api_key = st.text_input(
        "Enter your OpenAI API key to use MediGuide AI",
        type="password",
        placeholder="sk-...",
        help="Used only for this session in your browser -- never stored or logged by this app.",
    )
    st.caption("Don't have a key? Create one at platform.openai.com/api-keys")

    st.subheader("\u2699\uFE0F Model configuration")
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
    cache_choice = st.selectbox(
        "Caching", CACHE_OPTIONS, index=0,
        help="In-memory: fastest, cleared on restart. SQLite: persists across restarts.",
    )

    st.subheader("\U0001F310 Language")
    language = st.selectbox("Answer language", LANGUAGE_OPTIONS, index=0)

    apply_cache_choice(cache_choice)

    st.divider()
    st.caption("This app is for education only and is not a medical device.")

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title(f"{APP_NAME} \U0001FA7A")
st.info(
    "Fill in the form below. MediGuide AI will provide general, educational "
    "guidance -- always confirm with a real doctor.",
    icon="\u2139\uFE0F",
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Patient age", placeholder="e.g. 32")
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        duration = st.selectbox("Duration of symptoms", DURATION_OPTIONS)
    with col2:
        severity = st.slider("Severity (1 = mild, 10 = severe)", 1, 10, 3)
        symptoms = st.multiselect("Symptoms", SYMPTOM_OPTIONS)
        extra_symptoms = st.text_input("Other symptoms (free text, optional)")

    existing_conditions = st.text_area(
        "Existing medical conditions", placeholder="e.g. asthma, diabetes... (or 'none')"
    )
    medications = st.text_area(
        "Current medications", placeholder="e.g. metformin... (or 'none')"
    )
    notes = st.text_area("Additional notes", placeholder="Anything else you'd like to mention")

    submitted = st.form_submit_button("\U0001F50D Get Guidance", use_container_width=True)

# ---------------------------------------------------------------------------
# ON SUBMIT
# ---------------------------------------------------------------------------
if submitted:
    api_key = resolve_api_key(user_api_key)

    if not api_key:
        st.error("Please enter an OpenAI API key in the sidebar to continue.", icon="\U0001F6AB")
        st.stop()

    if not symptoms and not extra_symptoms.strip():
        st.warning("Please enter at least one symptom before submitting.", icon="\u26A0\uFE0F")
        st.stop()

    symptom_summary = format_symptom_summary(symptoms, extra_symptoms)

    inputs = {
        "age": age or "Not specified",
        "gender": gender,
        "symptoms": symptom_summary,
        "duration": duration,
        "severity": severity,
        "existing_conditions": existing_conditions or "None reported",
        "medications": medications or "None reported",
        "notes": notes or "None",
        "language": language,
    }

    try:
        llm = build_llm(api_key=api_key, model_name=model_name, temperature=temperature, streaming=False)
        chain = build_assessment_chain(llm)

        with st.spinner("Analyzing symptoms..."):
            raw_text = run_assessment_chain(chain, inputs)

        data, error = safe_parse_json(raw_text)

        if error:
            st.error("The AI response could not be parsed as valid JSON.", icon="\U0001F6AB")
            with st.expander("Show raw model output (debug)"):
                st.code(raw_text)
        else:
            urgency = str(data.get("urgency_level", "UNKNOWN")).upper()

            if urgency == "EMERGENCY":
                st.error(
                    "\U0001F6A8 **EMERGENCY:** This may be urgent. Please seek emergency "
                    "medical care immediately or call your local emergency number.",
                    icon="\U0001F6A8",
                )
            elif urgency == "HIGH":
                st.warning("\u26A0\uFE0F High urgency: please consult a healthcare professional soon.", icon="\u26A0\uFE0F")
            elif urgency == "MEDIUM":
                st.info("A healthcare professional should be consulted.", icon="\u2139\uFE0F")
            else:
                st.success("Low urgency: general self-care and monitoring likely appropriate.", icon="\u2705")

            m1, m2, m3 = st.columns(3)
            m1.metric("Urgency level", urgency_badge(urgency))
            m2.metric("Reported severity", f"{severity}/10")
            m3.metric("Symptom duration", duration)

            tab1, tab2, tab3, tab4 = st.tabs(
                ["\U0001F4CB Summary", "\U0001FA7B Possible conditions", "\u2705 Next steps", "\U0001F5E3\uFE0F For your doctor"]
            )

            with tab1:
                st.subheader("Patient symptom summary")
                st.write(f"**Symptoms:** {symptom_summary}")
                st.write(f"**AI summary:** {data.get('summary', 'N/A')}")

                st.subheader("Live narrative")
                streaming_llm = build_llm(
                    api_key=api_key, model_name=model_name, temperature=temperature, streaming=True
                )
                st.write_stream(stream_narrative(streaming_llm, inputs))

            with tab2:
                conditions = data.get("possible_conditions", [])
                if conditions:
                    for c in conditions:
                        with st.expander(f"\U0001FA7B {c.get('name', 'Unnamed condition')}"):
                            st.write(c.get("reason", "No reason provided."))
                else:
                    st.write("No possible conditions were suggested.")
                st.caption("These are for educational purposes only, not a diagnosis.")

            with tab3:
                st.subheader("Recommended next steps")
                for step in data.get("recommended_next_steps", []):
                    st.write(f"- {step}")

                st.subheader("\u26A0\uFE0F Warning signs requiring immediate attention")
                for sign in data.get("warning_signs", []):
                    st.write(f"- {sign}")

            with tab4:
                st.subheader("Questions to ask your healthcare professional")
                for q in data.get("questions_for_doctor", []):
                    st.write(f"- {q}")

            st.divider()
            st.warning(
                "Reminder: MediGuide AI is an educational prototype, not a licensed "
                "medical professional. Always confirm any guidance with a real doctor, "
                "and seek emergency care for severe or worsening symptoms.",
                icon="\u26A0\uFE0F",
            )

    except Exception as e:
        st.error(f"Something went wrong while contacting the AI model: {e}", icon="\U0001F6AB")
