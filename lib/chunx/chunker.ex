defmodule Chunx.Chunker do
  @moduledoc """
  Defines the shared callback types for chunkers.

  Chunker input must be valid UTF-8. Invalid text is returned as
  `{:error, {:invalid_text, :invalid_utf8}}` before tokenization begins.
  """

  alias Chunx.{Chunk, SentenceChunk, Tokenizer}

  @type embedding_fun :: ([String.t()] -> [Nx.Tensor.t()])
  @type chunk_result :: {:ok, [Chunk.t()] | [SentenceChunk.t()]} | {:error, term()}

  @doc """
  Splits text using a tokenizer.

  Semantic chunkers take an embedding function instead of options as the third
  argument and accept options as a fourth argument.

  ## Parameters
    * `text` - The text to chunk
    * `tokenizer` - The tokenizer to use
    * `opts_or_embedding_fun` - Options specific to the chunking strategy, or
      the embedding function used by a semantic chunker
  """
  @callback chunk(
              text :: String.t(),
              tokenizer :: Tokenizer.t(),
              opts_or_embedding_fun :: keyword() | embedding_fun()
            ) :: chunk_result()

  @callback chunk(
              text :: String.t(),
              tokenizer :: Tokenizer.t(),
              embedding_fun(),
              opts :: keyword()
            ) :: chunk_result()

  @optional_callbacks chunk: 4
end
