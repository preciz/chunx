defmodule Chunx.Chunk do
  @moduledoc """
  Struct representing a text chunk with metadata.

  ## Fields
    * `:text` - The text content of the chunk
    * `:start_byte` - The starting byte offset of the chunk in the original text
    * `:end_byte` - The ending byte offset of the chunk in the original text
    * `:token_count` - The number of content tokens in the chunk, excluding
      tokenizer entries without a byte span
    * `:embedding` - The embedding vector for the chunk (optional)
  """

  @enforce_keys [:text, :start_byte, :end_byte, :token_count]
  defstruct [:text, :start_byte, :end_byte, :token_count, :embedding]

  @type t :: %__MODULE__{
          text: String.t(),
          start_byte: non_neg_integer(),
          end_byte: non_neg_integer(),
          token_count: non_neg_integer(),
          embedding: Nx.Tensor.t() | nil
        }

  @doc "Creates a chunk. Byte offsets use a half-open range."
  @spec new(
          String.t(),
          non_neg_integer(),
          non_neg_integer(),
          non_neg_integer(),
          Nx.Tensor.t() | nil
        ) :: t()
  def new(text, start_byte, end_byte, token_count, embedding \\ nil)
      when is_binary(text) and
             is_integer(start_byte) and start_byte >= 0 and
             is_integer(end_byte) and end_byte >= start_byte and
             is_integer(token_count) and token_count >= 0 do
    if String.valid?(text) do
      %__MODULE__{
        text: text,
        start_byte: start_byte,
        end_byte: end_byte,
        token_count: token_count,
        embedding: embedding
      }
    else
      raise ArgumentError, "text must be valid UTF-8"
    end
  end
end
