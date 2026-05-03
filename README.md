# UCSC AI Agent Course

This github repo is for the course. It includes resources and examples.


## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or
   pip install .
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Get a Google API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file



## Project Structure

```
├── main.py                     # Main example script
├── tools/
│   └── gemini/
│       └── gemini.py           # Gemini tool example and CLI
├── tools/tool_functions.py     # Example tool functions
├── pyproject.toml              # Project configuration
├── env.example                 # Environment variables template
└── README.md                   # This file
```

## Requirements

- Python 3.14+
- Google GenAI API key
- Dependencies listed in `pyproject.toml`


### Notes
git branch -M main
git remote add origin git@github.com:hienluu/ucsc-ai-agent-course.git
git push -u origin main