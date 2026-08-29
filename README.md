# MediGuide AI

An educational Streamlit + LangChain prototype that gathers basic patient
symptom information and produces structured, safety-focused general
guidance using an OpenAI chat model.

> **Not a medical device.** MediGuide AI never gives a confirmed diagnosis.
> It always tells the user to consult a licensed healthcare professional,
> and to seek emergency care for urgent symptoms.

## Project structure

```
mediguide_ai/
|-- app.py                 Streamlit UI (run this)
|-- requirements.txt
|-- .env.example
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py          settings + form options + API key resolution
|   |-- prompts.py         PromptTemplate + ChatPromptTemplate + JSON schema
|   |-- chains.py          ChatOpenAI, LLMChain, message demo, streaming
|   |-- cache_manager.py   in-memory + SQLite caching
|   |-- utils.py           safe JSON parsing + helpers
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. (Optional, for local dev) copy `.env.example` to `.env` and paste a real
   OpenAI key:
   ```bash
   cp .env.example .env
   ```
   You normally won't need this: the app has a sidebar field where any
   visitor can paste their own OpenAI API key at runtime. That key is used
   only for that browser session and is never written to disk or logged.
   The `.env` key is only used as a fallback if the sidebar field is left
   empty.

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`),
paste an OpenAI API key into the sidebar, fill in the symptom form, and
click **Get Guidance**.

## Caching: InMemoryCache vs SQLiteCache

Both are demonstrated in `src/cache_manager.py` and can be switched from
the sidebar ("No caching" / "In-memory cache" / "SQLite cache"). A cache is
registered once with `set_llm_cache(...)`; LangChain then checks it
automatically before every model call, so submitting the exact same form
twice should be visibly faster the second time.

| | InMemoryCache | SQLiteCache |
|---|---|---|
| Stored in | RAM (memory) | A file on disk (`.mediguide_cache.db`) |
| Speed | Fastest | Fast, slightly slower than memory |
| Survives app restart? | No | Yes |
| Best for | A single running session | Reusing results across sessions/restarts |

## LangChain version compatibility notes

This project targets **current** LangChain (0.3+), `langchain-core` (0.3+),
and `langchain-openai` (0.2+) -- not the older pre-0.1 API. Key points:

- `ChatOpenAI` is imported from `langchain_openai`, not `langchain.chat_models`.
- `SystemMessage` / `HumanMessage` / `AIMessage` come from `langchain_core.messages`.
- `PromptTemplate` and `ChatPromptTemplate` come from `langchain_core.prompts`.
- `set_llm_cache` comes from `langchain_core.globals`; `InMemoryCache` and
  `SQLiteCache` come from `langchain_community.cache`.
- `LLMChain` (from `langchain.chains`) is used to satisfy the assignment's
  explicit LLMChain requirement; it's called with `.invoke(inputs)`, which
  returns a dict whose text is under the `"text"` key.
- Streaming uses `llm.stream(messages)` on a `ChatOpenAI` instance built
  with `streaming=True`, yielded into `st.write_stream(...)`.

`src/chains.py` already handles this automatically: it tries to import
`LLMChain` from the base `langchain` package, and if that package isn't
installed (a common `ModuleNotFoundError: No module named 'langchain.chains'`
if you only installed `langchain-core`/`langchain-openai`/`langchain-community`),
it silently falls back to the LCEL pipeline `ASSESSMENT_PROMPT_TEMPLATE | llm`.
`run_assessment_chain()` reads the result correctly either way, so the app
works whether or not `langchain` itself is installed. To get the literal
`LLMChain` demo, just run `pip install langchain` (already in
`requirements.txt`).

## Testing scenarios

See the assignment brief for the full list (mild cold, moderate flu,
emergency chest pain, repeat-submission caching check, empty-symptom
validation, and a non-English language run).

## Disclaimer

This project is for education only. It is not a medical device and must
not be used for real diagnosis or treatment.
