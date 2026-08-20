# Chunx

[![test](https://github.com/preciz/chunx/actions/workflows/test.yml/badge.svg)](https://github.com/preciz/chunx/actions/workflows/test.yml)

Chunx splits text by tokens, words, sentences, document structure, or semantic
similarity. It is an Elixir implementation inspired by
[Chonkie](https://github.com/chonkie-ai/chonkie).

## Installation

Add Chunx to `mix.exs`:

```elixir
def deps do
  [
    {:chunx, "~> 0.2.0"}
  ]
end
```

## Usage

All chunkers require a tokenizer. They accept a `Tokenizers.Tokenizer` or a
custom adapter implementing the `Chunx.Tokenizer` behaviour.

```elixir
alias Chunx.Chunker.Token

{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
{:ok, chunks} = Token.chunk("Text to split", tokenizer, chunk_size: 128)
```

Each returned chunk contains its text, half-open byte offsets into the original
text, and its content-token count. Sentence and Semantic return
`Chunx.SentenceChunk` structs; the other chunkers return `Chunx.Chunk` structs.

### Chunkers

| Module | Splitting unit | Overlap |
| --- | --- | --- |
| `Chunx.Chunker.Token` | Token offsets | Token count or fraction |
| `Chunx.Chunker.Word` | Whole words | Token count or fraction |
| `Chunx.Chunker.Sentence` | Whole sentences | Whole sentences within a token budget |
| `Chunx.Chunker.Recursive` | Configured structural levels, then tokens | None |
| `Chunx.Chunker.Semantic` | Sentence-embedding similarity | None |

See the [API documentation](https://hexdocs.pm/chunx/) for each module's options
and size-limit exceptions.

Semantic chunking also requires a function that returns one `Nx.Tensor` for
each input string:

```elixir
alias Chunx.Chunker.Semantic

embedding_fun = &MyApp.Embeddings.embed/1

{:ok, chunks} =
  Semantic.chunk("Text to split", tokenizer, embedding_fun,
    chunk_size: 128,
    threshold: :auto
  )
```

The repository also includes a
[non-semantic example](https://github.com/preciz/chunx/blob/v0.2.0/examples/demo.exs).

## Testing

Run the regular suite:

```bash
mix test
```

Embedding integration tests use
`sentence-transformers/all-MiniLM-L6-v2`. They download and run the model, so
they are excluded by default:

```bash
mix test --only integration
```

Use `mix test --include integration` to run both suites together.

## License

[MIT](LICENSE)
