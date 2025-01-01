defmodule Chunx.Chunk do
  @moduledoc """
  Struct representing a text chunk with metadata.

  ## Fields
    * `:text` - The text content of the chunk
    * `:start_byte` - The starting index of the chunk in the original text
    * `:end_byte` - The ending index of the chunk in the original text
    * `:token_count` - The number of tokens in the chunk
    * `:embedding` - The embedding vector for the chunk (optional)
  """

  @enforce_keys [:text, :start_byte, :end_byte, :token_count]
  defstruct [:text, :start_byte, :end_byte, :token_count, :embedding]

  @type t :: %__MODULE__{
          text: String.t(),
          start_byte: non_neg_integer(),
          end_byte: non_neg_integer(),
          token_count: pos_integer(),
          embedding: Nx.Tensor.t() | nil
        }

  @spec new(String.t(), non_neg_integer(), non_neg_integer(), pos_integer(), Nx.Tensor.t() | nil) ::
          t()
  def new(text, start_byte, end_byte, token_count, embedding \\ nil)
      when is_binary(text) and
             is_integer(start_byte) and start_byte >= 0 and
             is_integer(end_byte) and end_byte >= start_byte and
             is_integer(token_count) and token_count > 0 do
    %__MODULE__{
      text: text,
      start_byte: start_byte,
      end_byte: end_byte,
      token_count: token_count,
      embedding: embedding
    }
  end
end
