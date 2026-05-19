"""
llm_query_cache.py
==================
Full implementation of the LLM Query Caching Flow using Groq free API.

Pipeline overview:
  1. Normalize the user query
  2. Check FAQ cache (Redis) → return instantly if found
  3. Check dynamic response cache (Redis) → return if found
  4. Call the LLM via Groq API
  5. Track how many times this query has been seen
  6. If seen >= 3 times, store the response in Redis with a 1-hour TTL
  7. Return the final response

Requirements:
    pip install redis groq python-dotenv

Get your FREE Groq API key at: https://console.groq.com

Environment variables (.env file or shell exports):
    REDIS_HOST       = localhost          (default)
    REDIS_PORT       = 6379              (default)
    REDIS_PASSWORD   = <your-password>   (optional, leave blank if none)
    GROQ_API_KEY     = gsk_...           (from console.groq.com)
    LLM_MODEL        = llama3-8b-8192    (default — free tier model)
    CACHE_TTL        = 3600              (seconds, default = 1 hour)
    CACHE_THRESHOLD  = 3                 (min query count before caching)

Free-tier Groq models you can use in LLM_MODEL:
    llama3-8b-8192        → fast, lightweight (recommended default)
    llama3-70b-8192       → more capable, slightly slower
    mixtral-8x7b-32768    → long context window
    gemma-7b-it           → Google Gemma
"""

import os
import hashlib
import logging

import redis
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from a .env file if present

# Redis connection settings
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD  = os.getenv("REDIS_PASSWORD", None)  # None = no auth

# Groq API settings
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3-8b-8192")  # Free tier default

# Caching behaviour
CACHE_TTL       = int(os.getenv("CACHE_TTL", 3600))      # 1 hour in seconds
CACHE_THRESHOLD = int(os.getenv("CACHE_THRESHOLD", 3))   # Cache after N hits

# Logging — change to DEBUG for verbose output
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

def get_redis_client() -> redis.Redis:
    """
    Create and return a Redis client.
    Raises redis.ConnectionError if the server is unreachable.
    """
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,   # Return str instead of bytes
    )
    client.ping()  # Fail fast if Redis is not available
    log.info("Redis connected → %s:%d", REDIS_HOST, REDIS_PORT)
    return client


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def get_groq_client() -> Groq:
    """
    Return a Groq client using your free API key.

    Get your key at: https://console.groq.com
    Set it as GROQ_API_KEY in your .env file or environment.
    """
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Get your free key at https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Normalize a user query to a consistent lowercase, stripped form.
    Ensures 'Hello World' and 'hello world ' map to the same Redis key.
    """
    return query.lower().strip()


def hash_query(query: str) -> str:
    """
    Return a short MD5 hex digest of the normalized query.
    Used as part of Redis keys to avoid whitespace / special-char issues.

    Example:
        hash_query("what is redis?") → "a3f2c1d4..."
    """
    return hashlib.md5(query.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Redis key builders
# ---------------------------------------------------------------------------

def faq_key(query: str) -> str:
    """Static FAQ entries.  Pattern: faq:{normalized_query}"""
    return f"faq:{query}"


def response_key(query_hash: str) -> str:
    """Cached LLM responses.  Pattern: response:{hash}"""
    return f"response:{query_hash}"


def count_key(query_hash: str) -> str:
    """Query frequency counter.  Pattern: count:{hash}"""
    return f"count:{query_hash}"


# ---------------------------------------------------------------------------
# Step 1 — FAQ cache lookup
# ---------------------------------------------------------------------------

def check_faq_cache(client: redis.Redis, query: str) -> str | None:
    """
    Look up the query in the static FAQ cache.

    FAQ entries are pre-loaded manually and never expire automatically —
    they are managed content, not computed LLM results.

    Returns the cached answer string, or None if no entry exists.
    """
    key = faq_key(query)
    answer = client.get(key)

    if answer:
        log.info("FAQ cache HIT  → key: %s", key)
        return answer

    log.debug("FAQ cache MISS → key: %s", key)
    return None


# ---------------------------------------------------------------------------
# Step 2 — Dynamic response cache lookup
# ---------------------------------------------------------------------------

def check_response_cache(client: redis.Redis, query_hash: str) -> str | None:
    """
    Look up the query hash in the dynamic response cache.

    These entries are created automatically after a query is seen
    >= CACHE_THRESHOLD times, and expire after CACHE_TTL seconds.

    Returns the cached response string, or None on a cache miss.
    """
    key = response_key(query_hash)
    cached = client.get(key)

    if cached:
        log.info("Response cache HIT  → key: %s", key)
        return cached

    log.debug("Response cache MISS → key: %s", key)
    return None


# ---------------------------------------------------------------------------
# Step 3 — Groq LLM call
# ---------------------------------------------------------------------------

def call_llm(groq_client: Groq, query: str) -> str:
    """
    Send the query to Groq and return the response text.

    Groq's API is OpenAI-compatible, so the interface is identical.
    The free tier supports generous rate limits for prototyping.

    You can extend the messages list to include:
        - A detailed system prompt
        - Conversation history (multi-turn chat)
        - Retrieved context (RAG)
    """
    log.info("Calling Groq LLM (model=%s) …", LLM_MODEL)

    completion = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.7,    # 0 = deterministic, 1 = more creative
        max_tokens=1024,    # Keep responses reasonably sized
    )

    response_text = completion.choices[0].message.content
    log.info("Groq response received (%d chars).", len(response_text))
    return response_text


# ---------------------------------------------------------------------------
# Step 4 — Query frequency tracking + conditional caching
# ---------------------------------------------------------------------------

def track_and_maybe_cache(
    client: redis.Redis,
    query_hash: str,
    response: str,
) -> int:
    """
    Increment the query counter and, if the count reaches CACHE_THRESHOLD,
    store the response in Redis with a TTL of CACHE_TTL seconds.

    Returns the updated query count.

    Redis INCR is atomic — concurrent requests for the same query
    are handled safely without application-level locking.
    """
    c_key = count_key(query_hash)
    r_key = response_key(query_hash)

    # Atomically increment (creates the key at 0 then adds 1 if absent)
    current_count = client.incr(c_key)
    log.info("Query count for hash %s → %d", query_hash, current_count)

    if current_count >= CACHE_THRESHOLD:
        # SETEX sets value + TTL in a single atomic operation
        client.setex(r_key, CACHE_TTL, response)
        log.info(
            "Response cached in Redis (TTL=%ds) → key: %s",
            CACHE_TTL,
            r_key,
        )
    else:
        log.info(
            "Query count %d < threshold %d — skipping cache.",
            current_count,
            CACHE_THRESHOLD,
        )

    return current_count


# ---------------------------------------------------------------------------
# Main pipeline — assembles all steps
# ---------------------------------------------------------------------------

def handle_query(query: str) -> dict:
    """
    Run the full query-caching pipeline and return a result dict:

        {
            "response":    str,          # The answer text
            "source":      str,          # "faq_cache" | "response_cache" | "llm"
            "query_count": int | None    # Query frequency (None for cache hits)
        }

    Raises:
        redis.ConnectionError  — if Redis is unreachable
        EnvironmentError       — if GROQ_API_KEY is missing
        groq.APIError          — on Groq API failures
    """

    # --- Initialize clients -----------------------------------------------
    redis_client = get_redis_client()
    groq_client  = get_groq_client()

    # --- Normalize --------------------------------------------------------
    normalized = normalize_query(query)
    query_hash = hash_query(normalized)
    log.info("Query → normalized: %r | hash: %s", normalized, query_hash)

    # --- Step 1: FAQ cache ------------------------------------------------
    faq_answer = check_faq_cache(redis_client, normalized)
    if faq_answer:
        return {
            "response":    faq_answer,
            "source":      "faq_cache",
            "query_count": None,   # FAQ hits bypass the frequency counter
        }

    # --- Step 2: Dynamic response cache -----------------------------------
    cached_response = check_response_cache(redis_client, query_hash)
    if cached_response:
        return {
            "response":    cached_response,
            "source":      "response_cache",
            "query_count": None,
        }

    # --- Step 3: Call Groq LLM -------------------------------------------
    llm_response = call_llm(groq_client, normalized)

    # --- Step 4: Track count + maybe cache --------------------------------
    count = track_and_maybe_cache(redis_client, query_hash, llm_response)

    return {
        "response":    llm_response,
        "source":      "llm",
        "query_count": count,
    }


# ---------------------------------------------------------------------------
# FAQ loader — admin utility
# ---------------------------------------------------------------------------

def load_faq_entries(faq_data: dict[str, str]) -> None:
    """
    Pre-load static FAQ entries into Redis.

    Call this once during app startup or as part of a deployment script.
    Entries do NOT expire — delete or update them manually.

    Args:
        faq_data: mapping of {question_string: answer_string}

    Example:
        load_faq_entries({
            "what is your return policy?": "You can return any item within 30 days.",
            "how do i reset my password?": "Click 'Forgot password' on the login page.",
        })
    """
    client = get_redis_client()

    for question, answer in faq_data.items():
        normalized_q = normalize_query(question)
        key = faq_key(normalized_q)
        client.set(key, answer)
        log.info("FAQ loaded → key: %s", key)

    log.info("Loaded %d FAQ entries into Redis.", len(faq_data))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Seed a couple of FAQ entries for the demo
    sample_faqs = {
        "what is your opening hours?": "We are open Monday–Friday, 9 AM to 6 PM.",
        "where are you located?":      "Our office is at 123 Main Street, Suite 4B.",
    }
    load_faq_entries(sample_faqs)

    # Accept a query from the command line, or use a default
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is Redis?"

    print(f"\nQuery      : {user_query!r}")
    print("─" * 50)

    result = handle_query(user_query)

    print(f"Source     : {result['source']}")
    print(f"Query count: {result['query_count']}")
    print(f"Response   :\n{result['response']}\n")
