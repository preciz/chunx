defmodule Chunx.Chunker.Token do
  @moduledoc """
  Implements token based chunking strategy.

  Splits text into overlapping chunks based on token count using the given tokenizer.
  """

  @behaviour Chunx.Chunker

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
    * `:chunk_size` - Maximum number of tokens per chunk (default: 512)
    * `:chunk_overlap` - Number of tokens (integer) or percentage (float between 0 and 1) to overlap between chunks (default: 0.25)

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
  @spec chunk(binary(), Tokenizers.Tokenizer.t(), chunk_opts()) ::
          {:ok, [Chunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    if String.trim(text) == "" do
      {:ok, []}
    else
      with {:ok, encoding} <- Tokenizers.Tokenizer.encode(tokenizer, text) do
        chunks =
          encoding
          |> Tokenizers.Encoding.get_offsets()
          |> reject_empty_offsets()
          |> chunk_text(text, config)

        {:ok, chunks}
      end
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

  defp reject_empty_offsets(offsets),
    do: Enum.reject(offsets, fn {start_pos, end_pos} -> start_pos == end_pos end)

  defp chunk_text([], _text, _config), do: []

  defp chunk_text(valid_offsets, text, %{chunk_size: size, chunk_overlap: overlap}) do
    offsets = List.to_tuple(valid_offsets)
    build_chunks(offsets, text, size, size - overlap, 0, tuple_size(offsets), [])
  end

  defp build_chunks(_offsets, _text, _size, _step, start, total, chunks)
       when start >= total,
       do: Enum.reverse(chunks)

  defp build_chunks(offsets, text, size, step, start, total, chunks) do
    end_position = min(start + size, total)
    {start_offset, _} = elem(offsets, start)
    {_, end_offset} = elem(offsets, end_position - 1)

    chunk =
      Chunx.Chunk.new(
        binary_part(text, start_offset, end_offset - start_offset),
        start_offset,
        end_offset,
        end_position - start
      )

    build_chunks(offsets, text, size, step, start + step, total, [chunk | chunks])
  end
end
