# Copy this file to key_param.py and fill in your own values.
# key_param.py is gitignored so your secrets stay local.

# MongoDB Atlas connection string.
# Get it from the Atlas UI: Connect -> Drivers -> "Add your connection string".
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net"

# Voyage AI API key, used for embeddings.
# Sign up / create a key at https://dash.voyageai.com/
VOYAGE_API_KEY = "<your-voyage-api-key>"

# LM Studio exposes an OpenAI-compatible local server.
# The API key is required by the client but ignored by LM Studio,
# so any non-empty string works. base_url points to the local server
# (default port is 1234) and model is the one you loaded in LM Studio.
LLM_API_KEY = "lm-studio"
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL = "<model-name-as-shown-in-lm-studio>"
