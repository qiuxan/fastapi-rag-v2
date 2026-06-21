# OpenAI Provider Mode Design

## Goal

Make the real OpenAI embedding and answer generation path clear, tested, and documented
without making local development or automated tests depend on network access or API
credits.

The app already has `OpenAIEmbeddingProvider` and `OpenAIAnswerGenerator`. This task
will make that OpenAI mode safer to use and easier to verify.

## Current Behavior

The app chooses providers in `build_rag_service(settings)`:

- no `OPENAI_API_KEY`: use `MockEmbeddingProvider` and `MockAnswerGenerator`
- with `OPENAI_API_KEY`: use `OpenAIEmbeddingProvider` and `OpenAIAnswerGenerator`

SQLite is now the default vector store in both modes.

## Design Decision

Keep mock mode as the default.

OpenAI mode should be opt-in through environment configuration:

```text
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

This keeps the tutorial cheap, deterministic, and runnable without external services.
When the user wants real behavior, setting `OPENAI_API_KEY` switches both embedding and
answer generation to OpenAI.

## Provider Responsibilities

### `OpenAIEmbeddingProvider`

Responsibilities:

- create an OpenAI SDK client with the configured API key
- call `client.embeddings.create(model=..., input=text)`
- return `response.data[0].embedding` as `list[float]`

It should not split text, store vectors, or know about FastAPI.

### `OpenAIAnswerGenerator`

Responsibilities:

- return `"No documents have been indexed yet."` when no sources are passed
- build context from retrieved source chunks
- call `client.chat.completions.create(...)`
- return `response.choices[0].message.content` as a string

It should not retrieve sources or access SQLite directly.

## Testing Strategy

Automated tests will not call the real OpenAI API.

Instead, provider tests will inject fake clients that look like the small part of the
OpenAI SDK we use. This proves our code sends the expected model/input/messages and
parses the expected response shape.

Test coverage:

- `OpenAIEmbeddingProvider` passes the configured model and input text to the embeddings
  API.
- `OpenAIEmbeddingProvider` returns the embedding vector from the response.
- `OpenAIAnswerGenerator` returns the no-documents message without making an API call
  when sources are empty.
- `OpenAIAnswerGenerator` sends retrieved context and question to the chat completion
  API.
- `OpenAIAnswerGenerator` returns the assistant message content.
- settings tests continue to prove that an API key switches the app into OpenAI mode.

## Manual Verification

Manual live verification will be documented but not automated:

1. create `.env` from `.env.example`
2. set `OPENAI_API_KEY`
3. optionally set model names
4. start the API
5. ingest a document
6. ask a question
7. confirm the answer is no longer prefixed with `"Mock answer"`

This live check can spend API credits, so it should be explicit.

## Documentation

Update `.env.example` and `README.md` to explain:

- mock mode is the default
- OpenAI mode starts when `OPENAI_API_KEY` is set
- OpenAI mode uses real API calls and can cost money
- SQLite still stores chunks locally
- changing embedding models after indexing existing chunks can cause vector dimension
  mismatches; for this tutorial, clear `data/rag.db` when changing embedding models

## Out Of Scope

This task will not add:

- streaming answers
- Responses API migration
- async OpenAI client
- retries beyond the OpenAI SDK defaults
- token counting
- cost tracking
- per-request provider selection
- UI for provider selection
- production-grade secret management
