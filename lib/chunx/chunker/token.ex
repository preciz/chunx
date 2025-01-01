defmodule Chunx.Chunker.Token do
  @moduledoc """
  Implements token based chunking strategy.

  Splits text into overlapping chunks based on token count using the given tokenizer.
  """

  @behaviour Chunx.Chunker

  @type chunk_opts :: [
          chunk_size: pos_integer(),
          chunk_overlap: pos_integer() | float()
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

    with {:ok, encoding} <- Tokenizers.Tokenizer.encode(tokenizer, text) do
      chunks =
        encoding
        |> Tokenizers.Encoding.get_offsets()
        |> get_valid_token_positions()
        |> chunk_text(text, config)

      {:ok, chunks}
    end
  end

  defp validate_config!(opts) do
    size = Keyword.fetch!(opts, :chunk_size)
    overlap = Keyword.fetch!(opts, :chunk_overlap)

    if size <= 0, do: raise(ArgumentError, "chunk_size must be positive")

    if is_integer(overlap) and (overlap < 0 or overlap >= size),
      do: raise(ArgumentError, "chunk_overlap must be less than chunk_size")

    if is_float(overlap) and (overlap < 0.0 or overlap >= 1.0),
      do: raise(ArgumentError, "chunk_overlap percentage must be less than 1")

    %{
      chunk_size: size,
      chunk_overlap: if(is_float(overlap), do: floor(overlap * size), else: overlap)
    }
  end

  defp get_valid_token_positions(offsets) do
    offsets
    |> Enum.with_index()
    |> Enum.reject(fn {{start_pos, end_pos}, _} -> start_pos == end_pos end)
  end

  defp chunk_text([], _text, _config), do: []

  defp chunk_text(valid_tokens, text, config) do
    valid_tokens
    |> calculate_chunk_boundaries(config)
    |> Enum.map(&create_chunk(valid_tokens, text, &1))
  end

  defp calculate_chunk_boundaries(valid_tokens, %{chunk_size: size, chunk_overlap: overlap}) do
    total_tokens = length(valid_tokens)
    step = size - overlap

    0..total_tokens//step
    |> Enum.map(fn start -> {start, min(start + size, total_tokens)} end)
    |> Enum.take_while(fn {start, end_} -> end_ > start end)
  end

  defp create_chunk(valid_tokens, text, {start_position, end_position}) do
    tokens = Enum.slice(valid_tokens, start_position, end_position - start_position)
    {{start_offset, _}, _} = hd(tokens)
    {{_, end_offset}, _} = List.last(tokens)

    text_slice = binary_part(text, start_offset, end_offset - start_offset)
    token_count = end_position - start_position

    Chunx.Chunk.new(text_slice, start_offset, end_offset, token_count, nil)
  end
end
