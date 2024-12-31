defmodule Chunx.SentenceChunk do
  @moduledoc """
  Struct representing a sentence chunk with metadata.

  ## Fields
    * `:text` - The text content of the chunk
    * `:start_index` - The starting index of the chunk in the original text
    * `:end_index` - The ending index of the chunk in the original text
    * `:token_count` - The number of tokens in the chunk
    * `:sentences` - List of Chunk structs in the chunk
  """

  @enforce_keys [:text, :start_index, :end_index, :token_count, :sentences]
  defstruct [:text, :start_index, :end_index, :token_count, :sentences]

  @type t :: %__MODULE__{
          text: String.t(),
          start_index: non_neg_integer(),
          end_index: non_neg_integer(),
          token_count: pos_integer(),
          sentences: [Chunk.t()]
        }

  @spec new(String.t(), non_neg_integer(), non_neg_integer(), pos_integer(), [Chunk.t()]) :: t()
  def new(text, start_index, end_index, token_count, sentences)
      when is_binary(text) and
             is_integer(start_index) and start_index >= 0 and
             is_integer(end_index) and end_index >= start_index and
             is_integer(token_count) and token_count > 0 and
             is_list(sentences) do
    %__MODULE__{
      text: text,
      start_index: start_index,
      end_index: end_index,
      token_count: token_count,
      sentences: sentences
    }
  end
end
