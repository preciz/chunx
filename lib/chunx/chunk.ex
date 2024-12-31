defmodule Chunx.Chunk do
  @moduledoc """
  Struct representing a text chunk with metadata.

  ## Fields
    * `:text` - The text content of the chunk
    * `:start_index` - The starting index of the chunk in the original text
    * `:end_index` - The ending index of the chunk in the original text
    * `:token_count` - The number of tokens in the chunk
    * `:embedding` - The embedding vector for the chunk (optional)
  """

  @enforce_keys [:text, :start_index, :end_index, :token_count]
  defstruct [:text, :start_index, :end_index, :token_count, :embedding]

  @type t :: %__MODULE__{
          text: String.t(),
          start_index: non_neg_integer(),
          end_index: non_neg_integer(),
          token_count: pos_integer(),
          embedding: Nx.Tensor.t() | nil
        }

  @spec new(String.t(), non_neg_integer(), non_neg_integer(), pos_integer(), Nx.Tensor.t() | nil) ::
          t()
  def new(text, start_index, end_index, token_count, embedding \\ nil)
      when is_binary(text) and
             is_integer(start_index) and start_index >= 0 and
             is_integer(end_index) and end_index >= start_index and
             is_integer(token_count) and token_count > 0 do
    %__MODULE__{
      text: text,
      start_index: start_index,
      end_index: end_index,
      token_count: token_count,
      embedding: embedding
    }
  end
end
