defmodule Chunx.Chunker do
  @moduledoc """
  Defines the interface for text chunking strategies.
  """

  alias Chunx.Chunk

  @doc """
  Splits text into chunks using the given tokenizer.

  ## Parameters
    * `text` - The text to chunk
    * `tokenizer` - The tokenizer to use
    * `opts` - Options specific to the chunking strategy
  """
  @callback chunk(text :: String.t(), tokenizer :: Tokenizers.Tokenizer.t(), opts :: keyword()) ::
              {:ok, [Chunk.t()]} | {:error, any()}
end
