defmodule Chunx.Chunker.Token do
  @moduledoc """
  Splits text at token offsets, with optional overlap.
  """

  @behaviour Chunx.Chunker

  alias Chunx.{Chunk, Helper, Tokenizer}

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          chunk_overlap: non_neg_integer() | float()
        ]

  @default_opts [
    chunk_size: 512,
    chunk_overlap: 0.25
  ]

  @doc """
  Splits text into overlapping chunks using the given tokenizer.

  ## Options
    * `:chunk_size` - Target maximum number of content tokens per chunk (default: 512).
      A byte-indivisible grapheme may contain more tokens and is kept intact.
    * `:chunk_overlap` - Overlap as a token count or a fraction in the range
      `[0.0, 1.0)` (default: `0.25`).

  ## Examples

      iex> {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("distilbert/distilbert-base-uncased")
      iex> Chunx.Chunker.Token.chunk("Some text to split", tokenizer, chunk_size: 3, chunk_overlap: 1)
      {
        :ok,
        [
          %Chunx.Chunk{end_byte: 12, start_byte: 0, text: "Some text to", token_count: 3},
          %Chunx.Chunk{end_byte: 18, start_byte: 10, text: "to split", token_count: 2}
        ]
      }
  """
  @spec chunk(String.t(), Tokenizer.t(), chunk_opts()) ::
          {:ok, [Chunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    with :ok <- Helper.validate_text(text), do: chunk_valid_text(text, tokenizer, config)
  end

  defp chunk_valid_text(text, tokenizer, config) do
    if String.trim(text) == "", do: {:ok, []}, else: chunk_nonempty_text(text, tokenizer, config)
  end

  defp chunk_nonempty_text(text, tokenizer, config) do
    with {:ok, offsets} <- Tokenizer.offsets(tokenizer, text) do
      chunk_text(offsets, text, tokenizer, config)
    end
  end

  defp validate_config!(opts) do
    size = Keyword.fetch!(opts, :chunk_size)
    overlap = Keyword.fetch!(opts, :chunk_overlap)

    validate_chunk_size!(size)

    %{
      chunk_size: size,
      chunk_overlap: normalize_overlap!(overlap, size)
    }
  end

  defp validate_chunk_size!(size) when is_integer(size) and size > 0, do: :ok
  defp validate_chunk_size!(_size), do: raise(ArgumentError, "chunk_size must be positive")

  defp normalize_overlap!(overlap, size)
       when is_integer(overlap) and overlap >= 0 and overlap < size,
       do: overlap

  defp normalize_overlap!(overlap, _size) when is_integer(overlap),
    do: raise(ArgumentError, "chunk_overlap must be less than chunk_size")

  defp normalize_overlap!(overlap, size)
       when is_float(overlap) and overlap >= 0.0 and overlap < 1.0,
       do: floor(overlap * size)

  defp normalize_overlap!(overlap, _size) when is_float(overlap),
    do: raise(ArgumentError, "chunk_overlap percentage must be less than 1")

  defp normalize_overlap!(_overlap, _size),
    do: raise(ArgumentError, "chunk_overlap must be an integer or float")

  defp chunk_text([], _text, _tokenizer, _config), do: {:ok, []}

  defp chunk_text(offsets, text, tokenizer, %{chunk_size: size, chunk_overlap: overlap}) do
    result =
      offsets
      |> Tokenizer.units()
      |> Tokenizer.pack(size, overlap)
      |> Enum.reduce_while([], fn units, chunks ->
        case create_chunk(units, text, tokenizer) do
          {:ok, chunk} -> {:cont, [chunk | chunks]}
          {:error, _reason} = error -> {:halt, error}
        end
      end)

    case result do
      {:error, _reason} = error -> error
      chunks -> {:ok, Enum.reverse(chunks)}
    end
  end

  defp create_chunk(units, text, tokenizer) do
    {start_offset, _, _} = hd(units)
    {_, end_offset, _} = List.last(units)
    chunk_text = binary_part(text, start_offset, end_offset - start_offset)

    with {:ok, token_count} <- Tokenizer.count(tokenizer, chunk_text) do
      {:ok, Chunk.new(chunk_text, start_offset, end_offset, token_count)}
    end
  end
end
