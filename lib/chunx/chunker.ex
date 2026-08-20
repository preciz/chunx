defmodule Chunx.Chunker do
  @moduledoc """
  Defines the interface for text chunking strategies.
  """

  alias Chunx.{Chunk, SentenceChunk, Tokenizer}

  @type embedding_fun :: ([String.t()] -> [Nx.Tensor.t()])
  @type chunk_result :: {:ok, [Chunk.t()] | [SentenceChunk.t()]} | {:error, term()}

  @doc """
  Splits text into chunks using the given tokenizer. Semantic chunkers receive
  an embedding function as the third argument and may receive options as a
  fourth argument.

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
