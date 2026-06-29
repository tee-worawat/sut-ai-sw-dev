# sut-ai-sw-dev

Course materials for prompt engineering, adapted to use Google's Gemini API (free tier) instead of OpenAI.

## Setup

1. Install dependencies:

```bash
pip install -r week_1/requirements.txt
```

2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

3. Create a `.env` file in the project root (copy from `.env.example`):

```
GOOGLE_API_KEY=your-api-key-here
```

4. Open and run the notebooks in `week_1/`. They import helper functions from `week_1/gemini_helpers.py`, which uses the `gemini-2.0-flash` model by default.
