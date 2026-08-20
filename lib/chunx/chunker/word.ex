defmodule Chunx.Chunker.Word do
  @moduledoc """
  Splits text at word boundaries, with optional overlap.
  """

  @behaviour Chunx.Chunker

  alias Chunx.{Chunk, Tokenizer}

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
    * `:chunk_size` - Target maximum content-token count (default: 512). A word
      that exceeds the target is kept intact.
    * `:chunk_overlap` - Overlap as a token count or a fraction in the range
      `[0.0, 1.0)` (default: `0.25`).

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
  @spec chunk(binary(), Tokenizer.t(), chunk_opts()) ::
          {:ok, [Chunk.t()]} | {:error, term()}
  def chunk(text, tokenizer, opts \\ []) when is_binary(text) do
    opts = Keyword.merge(@default_opts, opts)
    config = validate_config!(opts)

    if String.trim(text) == "" do
      {:ok, []}
    else
      text
      |> split_into_words()
      |> add_token_counts(tokenizer)
      |> then(fn
        {:ok, words} -> create_chunks(words, tokenizer, config)
        {:error, _reason} = error -> error
      end)
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
    result =
      Enum.reduce_while(words, {[], %{}}, &add_token_count(&1, &2, tokenizer))

    case result do
      {:error, _reason} = error -> error
      {words, _cache} -> {:ok, Enum.reverse(words)}
    end
  end

  defp add_token_count({word, _, _} = entry, {words, cache}, tokenizer) do
    case Map.fetch(cache, word) do
      {:ok, token_count} ->
        {:cont, {[{entry, token_count} | words], cache}}

      :error ->
        count_uncached_word(entry, words, cache, tokenizer)
    end
  end

  defp count_uncached_word({word, _, _} = entry, words, cache, tokenizer) do
    case Tokenizer.count(tokenizer, word) do
      {:ok, token_count} ->
        {:cont, {[{entry, token_count} | words], Map.put(cache, word, token_count)}}

      {:error, _reason} = error ->
        {:halt, error}
    end
  end

  defp create_chunks(
         words,
         tokenizer,
         %{chunk_size: chunk_size, chunk_overlap: chunk_overlap}
       ) do
    if Enum.all?(words, fn {_word, token_count} -> token_count == 0 end) do
      {:ok, []}
    else
      do_create_chunks(words, tokenizer, chunk_size, chunk_overlap)
    end
  end

  defp do_create_chunks(words, tokenizer, chunk_size, chunk_overlap) do
    result =
      Enum.reduce_while(words, {[], [], 0}, fn entry, state ->
        add_word(entry, state, tokenizer, chunk_size, chunk_overlap)
      end)

    case result do
      {chunks, current_chunk, _current_length} ->
        with {:ok, final_chunk} <- current_chunk |> Enum.reverse() |> create_chunk(tokenizer) do
          {:ok, Enum.reverse([final_chunk | chunks])}
        end

      {:error, _reason} = error ->
        error
    end
  end

  defp add_word(
         {_word, token_count} = entry,
         {chunks, current_chunk, current_length},
         tokenizer,
         chunk_size,
         chunk_overlap
       ) do
    if current_length + token_count <= chunk_size or current_chunk == [] do
      {:cont, {chunks, [entry | current_chunk], current_length + token_count}}
    else
      finish_word_chunk(
        entry,
        token_count,
        chunks,
        current_chunk,
        tokenizer,
        chunk_size,
        chunk_overlap
      )
    end
  end

  defp finish_word_chunk(
         entry,
         token_count,
         chunks,
         current_chunk,
         tokenizer,
         chunk_size,
         chunk_overlap
       ) do
    case current_chunk |> Enum.reverse() |> create_chunk(tokenizer) do
      {:ok, chunk} ->
        available_overlap = max(chunk_size - token_count, 0)

        {overlap, overlap_length} =
          calculate_overlap(current_chunk, min(chunk_overlap, available_overlap))

        {:cont, {[chunk | chunks], [entry | overlap], overlap_length + token_count}}

      {:error, _reason} = error ->
        {:halt, error}
    end
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

    with {:ok, token_count} <- Tokenizer.count(tokenizer, chunk_text) do
      {:ok, Chunk.new(chunk_text, start_byte, end_byte, token_count)}
    end
  end
end
