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
      words = split_into_words(text)
      lengths = get_word_token_counts(words, tokenizer)
      chunks = create_chunks(words, lengths, tokenizer, config)
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
    split_points =
      Regex.scan(~r/\s*\S+/, text, return: :index)
      |> Enum.map(fn [{start, length}] ->
        text_part = binary_part(text, start, length)
        {text_part, start, start + length}
      end)

    last = List.last(split_points)
    last_point = elem(last, 2)

    if last_point < byte_size(text) do
      remaining_length = byte_size(text) - last_point
      trailing = binary_part(text, last_point, remaining_length)
      split_points ++ [{trailing, last_point, byte_size(text)}]
    else
      split_points
    end
  end

  defp get_word_token_counts(words, tokenizer) do
    words
    |> Enum.reduce({%{}, []}, fn {word, _, _}, {cache, counts} ->
      case cache do
        %{^word => length} ->
          {cache, [length | counts]}

        _ ->
          {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, word)
          length = Tokenizers.Encoding.get_length(encoding)

          {Map.put(cache, word, length), [length | counts]}
      end
    end)
    |> elem(1)
    |> Enum.reverse()
  end

  defp create_chunks(words, lengths, tokenizer, config) do
    words_with_lengths = Enum.zip(words, lengths)

    {chunks, current_chunk, _current_length} =
      Enum.reduce(words_with_lengths, {[], [], 0}, fn {word, length},
                                                      {chunks, current_chunk, current_length} ->
        if current_length + length <= config.chunk_size or current_chunk == [] do
          {chunks, [{word, length} | current_chunk], current_length + length}
        else
          chunk = current_chunk |> Enum.reverse() |> create_chunk(tokenizer)

          available_overlap = max(config.chunk_size - length, 0)

          {overlap_chunk_reversed, overlap_length} =
            calculate_overlap(current_chunk, min(config.chunk_overlap, available_overlap))

          new_chunk = [{word, length} | overlap_chunk_reversed]
          new_length = overlap_length + length

          {[chunk | chunks], new_chunk, new_length}
        end
      end)

    final_chunk = current_chunk |> Enum.reverse() |> create_chunk(tokenizer)

    [final_chunk | chunks]
    |> Enum.reverse()
    |> Enum.reject(&is_nil/1)
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

  defp create_chunk([], _tokenizer), do: nil

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
