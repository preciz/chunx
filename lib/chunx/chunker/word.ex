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
          chunk_overlap: pos_integer() | float()
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
          %Chunx.Chunk{end_index: 12, start_index: 0, text: "Some text to", token_count: 3},
          %Chunx.Chunk{end_index: 18, start_index: 9, text: " to split", token_count: 2}
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
      chunks = create_chunks(words, lengths, text, config)
      {:ok, chunks}
    end
  end

  defp validate_config!(opts) do
    size = Keyword.fetch!(opts, :chunk_size)
    # Default to 0 if not provided
    overlap = Keyword.get(opts, :chunk_overlap, 0)

    if size <= 0, do: raise(ArgumentError, "chunk_size must be positive")

    if is_integer(overlap) and (overlap < 0 or overlap >= size) do
      raise(ArgumentError, "chunk_overlap must be less than chunk_size and non-negative")
    end

    if is_float(overlap) and (overlap < 0.0 or overlap >= 1.0) do
      raise(ArgumentError, "chunk_overlap percentage must be less than 1")
    end

    overlap_size = if is_float(overlap), do: floor(overlap * size), else: overlap
    %{chunk_size: size, chunk_overlap: overlap_size}
  end

  defp split_into_words(text) do
    split_points =
      Regex.scan(~r/\s*\S+/, text, return: :index)
      |> Enum.map(fn [{start, length}] ->
        text_part = binary_part(text, start, length)
        {start, text_part}
      end)

    last_point =
      case split_points do
        [] ->
          0

        points ->
          last = List.last(points)
          elem(last, 0) + byte_size(elem(last, 1))
      end

    if last_point < byte_size(text) do
      remaining_length = byte_size(text) - last_point
      split_points ++ [{last_point, binary_part(text, last_point, remaining_length)}]
    else
      split_points
    end
    |> Enum.map(&elem(&1, 1))
  end

  defp get_word_token_counts(words, tokenizer) do
    words
    |> Enum.reduce({%{}, []}, fn word, {cache, counts} ->
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

  defp create_chunks(words, lengths, text, config) do
    words_with_lengths =
      Enum.zip(words, lengths)
      |> Enum.with_index()

    {chunks, current_chunk, current_length, _} =
      Enum.reduce(words_with_lengths, {[], [], 0, 0}, fn {{word, length}, idx},
                                                         {chunks, current_chunk, current_length,
                                                          _} ->
        if current_length + length <= config.chunk_size do
          {chunks, current_chunk ++ [word], current_length + length, idx}
        else
          chunk = create_chunk(current_chunk, text, current_length)

          # Calculate overlap similar to Python version
          overlap_start_idx = max(0, idx - length(current_chunk))

          {overlap_words, overlap_length} =
            words_with_lengths
            |> Enum.slice(overlap_start_idx, idx - overlap_start_idx)
            |> Enum.reverse()
            |> Enum.reduce_while({[], 0}, fn {{w, l}, _}, {acc, len} ->
              if len + l <= config.chunk_overlap do
                {:cont, {[w | acc], len + l}}
              else
                {:halt, {acc, len}}
              end
            end)

          new_chunk = overlap_words ++ [word]
          new_length = overlap_length + length

          {chunks ++ [chunk], new_chunk, new_length, idx}
        end
      end)

    if current_chunk != [] do
      chunks ++ [create_chunk(current_chunk, text, current_length)]
    else
      chunks
    end
  end

  defp create_chunk(words, text, token_count) do
    chunk_text = Enum.join(words)
    {start_index, length} = :binary.match(text, chunk_text)

    Chunk.new(chunk_text, start_index, start_index + length, token_count)
  end
end
