defmodule Chunx do
  @moduledoc """
  Text chunking strategies.

    * `Chunx.Chunker.Token` splits at token offsets.
    * `Chunx.Chunker.Word` keeps words intact.
    * `Chunx.Chunker.Sentence` keeps sentences intact and supports overlap.
    * `Chunx.Chunker.Recursive` tries structural boundaries before tokens.
    * `Chunx.Chunker.Semantic` splits where sentence similarity decreases.
  """
end
