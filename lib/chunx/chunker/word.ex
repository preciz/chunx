defmodule Chunx.Chunker.Word do
  @moduledoc """
  Implements word based chunking strategy.

  Splits text into overlapping chunks based on words while
  respecting token limits.
  """

  @behaviour Chunx.Chunker

  alias Chunx.Chunk

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          chunk_overlap: non_neg_integer() | float()
        ]

  @default_opts [
    chunk_size: 512,
    chunk_overlap: 0.25
  ]

  @doc """
  Splits text into overlapping chunks using word boundaries.

  ## Options
    * `:chunk_size` - Maximum number of tokens per chunk (default: 512)
    * `:chunk_overlap` - Number of tokens (integer) or percentage (float between 0 and 1) to overlap between chunks (default: 0.25)

  ## Examples

      iex> {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
      iex> Chunx.Chunker.Word.chunk("Some text to split", tokenizer, chunk_size: 3, chunk_overlap: 1)
      {
        :ok,
        [
          %Chunx.Chunk{end_byte: 12, start_byte: 0, text: "Some text to", token_count: 3},
          %Chunx.Chunk{end_byte: 18, start_byte: 9, text: " to split", token_count: 2}
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
      chunks =
        text
        |> split_into_words()
        |> add_token_counts(tokenizer)
        |> create_chunks(tokenizer, config)

      {:ok, chunks}
    end
  end

  defp validate_config!(opts) do
    size = Keyword.fetch!(opts, :chunk_size)
    overlap = Keyword.fetch!(opts, :chunk_overlap)

    validate_chunk_size!(size)

    %{chunk_size: size, chunk_overlap: normalize_overlap!(overlap, size)}
  end

  defp validate_chunk_size!(size) when is_integer(size) and size > 0, do: :ok
  defp validate_chunk_size!(_size), do: raise(ArgumentError, "chunk_size must be positive")

  defp normalize_overlap!(overlap, size)
       when is_integer(overlap) and overlap >= 0 and overlap < size,
       do: overlap

  defp normalize_overlap!(overlap, _size) when is_integer(overlap),
    do: raise(ArgumentError, "chunk_overlap must be less than chunk_size and non-negative")

  defp normalize_overlap!(overlap, size)
       when is_float(overlap) and overlap >= 0.0 and overlap < 1.0,
       do: floor(overlap * size)

  defp normalize_overlap!(overlap, _size) when is_float(overlap),
    do: raise(ArgumentError, "chunk_overlap percentage must be less than 1")

  defp normalize_overlap!(_overlap, _size),
    do: raise(ArgumentError, "chunk_overlap must be an integer or float")

  defp split_into_words(text) do
    {split_points, last_point} =
      Regex.scan(~r/\s*\S+/, text, return: :index)
      |> Enum.reduce({[], 0}, fn [{start, length}], {split_points, _last_point} ->
        text_part = binary_part(text, start, length)
        end_byte = start + length
        {[{text_part, start, end_byte} | split_points], end_byte}
      end)

    split_points =
      if last_point < byte_size(text) do
        trailing = binary_part(text, last_point, byte_size(text) - last_point)
        [{trailing, last_point, byte_size(text)} | split_points]
      else
        split_points
      end

    Enum.reverse(split_points)
  end

  defp add_token_counts(words, tokenizer) do
    {words, _cache} =
      Enum.map_reduce(words, %{}, fn {word, _, _} = word_with_offsets, cache ->
        case cache do
          %{^word => token_count} ->
            {{word_with_offsets, token_count}, cache}

          _ ->
            {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, word)
            token_count = Tokenizers.Encoding.get_length(encoding)

            {{word_with_offsets, token_count}, Map.put(cache, word, token_count)}
        end
      end)

    words
  end

  defp create_chunks(
         words,
         tokenizer,
         %{chunk_size: chunk_size, chunk_overlap: chunk_overlap}
       ) do
    {chunks, current_chunk, _current_length} =
      Enum.reduce(words, {[], [], 0}, fn {_word, token_count} = entry,
                                         {chunks, current_chunk, current_length} ->
        if current_length + token_count <= chunk_size or current_chunk == [] do
          {chunks, [entry | current_chunk], current_length + token_count}
        else
          chunk = current_chunk |> Enum.reverse() |> create_chunk(tokenizer)

          available_overlap = max(chunk_size - token_count, 0)

          {overlap_chunk_reversed, overlap_length} =
            calculate_overlap(current_chunk, min(chunk_overlap, available_overlap))

          new_chunk = [entry | overlap_chunk_reversed]
          new_length = overlap_length + token_count

          {[chunk | chunks], new_chunk, new_length}
        end
      end)

    final_chunk = current_chunk |> Enum.reverse() |> create_chunk(tokenizer)

    [final_chunk | chunks]
    |> Enum.reverse()
  end

  defp calculate_overlap(current_chunk, chunk_overlap) do
    {overlap_chunk, overlap_length} =
      current_chunk
      |> Enum.reduce_while({[], 0}, fn {_, l} = item, {acc, len} ->
        if len + l <= chunk_overlap do
          {:cont, {[item | acc], len + l}}
        else
          {:halt, {acc, len}}
        end
      end)

    {Enum.reverse(overlap_chunk), overlap_length}
  end

  defp create_chunk(words, tokenizer) do
    {{_, start_byte, _}, _} = hd(words)
    {{_, _, end_byte}, _} = List.last(words)
    chunk_text = Enum.map_join(words, fn {{word, _, _}, _} -> word end)
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, chunk_text)

    Chunk.new(
      chunk_text,
      start_byte,
      end_byte,
      Tokenizers.Encoding.get_length(encoding)
    )
  end
end
