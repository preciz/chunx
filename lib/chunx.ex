defmodule Chunx do
  @moduledoc """
  Chunk text with different chunking strategies.

  Available chunking strategies:
  - `Chunx.Chunker.Token` - Splits text into overlapping chunks based on token count
  - `Chunx.Chunker.Word` - Splits text into overlapping chunks based on word boundaries
  - `Chunx.Chunker.Sentence` - Splits text into overlapping chunks based on sentence boundaries
  - `Chunx.Chunker.Semantic` - Splits text into overlapping chunks based on semantic similarity
  """
end
