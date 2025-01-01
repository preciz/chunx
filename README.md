# Chunx

Chunx is an Elixir library for splitting text into meaningful chunks using various strategies. It's particularly useful for processing large texts for LLMs, semantic search, and other NLP tasks.

## Credit

This library is based on [chonkie-ai/chonkie](https://github.com/chonkie-ai/chonkie)

## Features

- Multiple chunking strategies:
  - Token-based chunking
  - Word-based chunking
  - Sentence-based chunking
  - Semantic chunking with embeddings

- Configurable options for each strategy
- Support for overlapping chunks
- Token count tracking
- Embedding support

## Installation

Add `chunx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:chunx, github: "preciz/chunx"}
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

## Configuration

Each chunking strategy accepts various options to customize the chunking behavior:

- `chunk_size`: Maximum number of tokens per chunk
- `chunk_overlap`: Number of tokens or percentage to overlap between chunks
- `min_sentences`: Minimum number of sentences per chunk (for sentence-based)
- `threshold`: Similarity threshold for semantic chunking
- And more...

See the documentation for each chunker module for detailed configuration options.

## License

[MIT License](LICENSE)

