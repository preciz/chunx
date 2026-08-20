# Chunx

[![test](https://github.com/preciz/chunx/actions/workflows/test.yml/badge.svg)](https://github.com/preciz/chunx/actions/workflows/test.yml)

Chunx is an Elixir library for splitting text into meaningful chunks using various strategies. It's particularly useful for processing large texts for LLMs, semantic search, and other NLP tasks.

## Credit

This library is based on [chonkie-ai/chonkie](https://github.com/chonkie-ai/chonkie)

## Features

- Multiple chunking strategies:
  - Token-based chunking
  - Word-based chunking
  - Sentence-based chunking
  - Semantic chunking with embeddings
  - Recursive chunking using structural boundaries

- Configurable options for each strategy
- Support for overlapping chunks
- Token count tracking
- Embedding support

## Installation

Add `chunx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:chunx, "~> 0.2.0"}
  ]
end
```

## Usage

### Token-based Chunking

```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
{:ok, chunks} = Chunx.Chunker.Token.chunk("Your text here", tokenizer, chunk_size: 512)
```

### Word-based Chunking

```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
{:ok, chunks} = Chunx.Chunker.Word.chunk("Your text here", tokenizer, chunk_size: 512)
```

### Sentence-based Chunking

```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
{:ok, chunks} = Chunx.Chunker.Sentence.chunk("Your text here", tokenizer)
```

### Semantic Chunking

```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")

# The embedding function must return a list of Nx.Tensor.t()
embedding_fn = fn texts ->
  # Your embedding function here
end

{:ok, chunks} = Chunx.Chunker.Semantic.chunk("Your text here", tokenizer, embedding_fn)
```

### Recursive Chunking

```elixir
{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
{:ok, chunks} = Chunx.Chunker.Recursive.chunk("Your text here", tokenizer)
```

Recursive chunking tries paragraphs, sentences, punctuation, whitespace, and
finally token boundaries until every chunk fits within the configured size.

## Configuration

Each chunking strategy accepts various options to customize the chunking behavior:

- `chunk_size`: Maximum number of content tokens per chunk
- `chunk_overlap`: Number or proportion of content tokens shared by consecutive chunks
- `min_sentences_per_chunk`: Minimum number of sentences per sentence-based chunk
- `min_sentences`: Minimum number of sentences per semantic chunk
- `threshold`: Similarity threshold for semantic chunking
- And more...

See the documentation for each chunker module for detailed configuration options.

## Testing

```elixir
# Run the test suite
mix test
```

Real-model embedding integration tests are excluded by default because they
download and run a Hugging Face model. Enable them explicitly with:

```bash
mix test --include integration
```

To run only the integration tests, use `mix test --only integration`.

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Override it with
`CHUNX_EMBEDDING_MODEL`, provided the model is supported by Bumblebee's text
embedding serving and `Tokenizers.Tokenizer.from_pretrained/1`.

## License

[MIT License](LICENSE)
