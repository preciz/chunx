defmodule Chunx.Tokenizer do
  @moduledoc false

  @type offset :: {non_neg_integer(), non_neg_integer()}
  @type unit :: {non_neg_integer(), non_neg_integer(), pos_integer()}
  @type t :: Tokenizers.Tokenizer.t() | {module(), term()}

  @callback offsets(state :: term(), text :: binary()) ::
              {:ok, [offset()]} | {:error, term()}

  @spec offsets(t(), binary()) :: {:ok, [offset()]} | {:error, term()}
  def offsets(%Tokenizers.Tokenizer{} = tokenizer, text) do
    with {:ok, encoding} <- Tokenizers.Tokenizer.encode(tokenizer, text) do
      encoding
      |> Tokenizers.Encoding.get_offsets()
      |> content_offsets(text)
    end
  end

  def offsets({module, state}, text) when is_atom(module) do
    case module.offsets(state, text) do
      {:ok, offsets} -> content_offsets(offsets, text)
      {:error, _reason} = error -> error
      response -> {:error, {:invalid_tokenizer_response, response}}
    end
  end

  @spec count(t(), binary()) :: {:ok, non_neg_integer()} | {:error, term()}
  def count(%Tokenizers.Tokenizer{} = tokenizer, text) do
    with {:ok, encoding} <- Tokenizers.Tokenizer.encode(tokenizer, text) do
      count =
        encoding
        |> Tokenizers.Encoding.get_offsets()
        |> Enum.count(fn {start_byte, end_byte} -> start_byte < end_byte end)

      {:ok, count}
    end
  end

  def count(tokenizer, text) do
    with {:ok, offsets} <- offsets(tokenizer, text) do
      {:ok, length(offsets)}
    end
  end

  @spec units([offset()]) :: [unit()]
  def units(offsets) do
    offsets
    |> Enum.reduce([], &merge_offset/2)
    |> Enum.reverse()
  end

  @spec pack([unit()], pos_integer(), non_neg_integer()) :: [[unit()]]
  def pack([], _chunk_size, _overlap), do: []

  def pack(units, chunk_size, 0) do
    {groups, current, _token_count} =
      Enum.reduce(units, {[], [], 0}, fn {_, _, unit_count} = unit,
                                         {groups, current, token_count} ->
        if current == [] or token_count + unit_count <= chunk_size do
          {groups, [unit | current], token_count + unit_count}
        else
          {[Enum.reverse(current) | groups], [unit], unit_count}
        end
      end)

    Enum.reverse([Enum.reverse(current) | groups])
  end

  def pack(units, chunk_size, overlap) do
    unit_tuple = List.to_tuple(units)

    token_units =
      if Enum.all?(units, fn {_, _, count} -> count == 1 end) do
        nil
      else
        units
        |> Enum.with_index()
        |> Enum.flat_map(fn {{_, _, count}, index} -> List.duplicate(index, count) end)
        |> List.to_tuple()
      end

    total = if token_units, do: tuple_size(token_units), else: tuple_size(unit_tuple)

    pack_windows(
      unit_tuple,
      token_units,
      total,
      chunk_size,
      chunk_size - overlap,
      0,
      nil,
      []
    )
  end

  defp content_offsets(offsets, text) when is_list(offsets) do
    boundaries = grapheme_boundaries(text, 0, MapSet.new([0]))
    text_size = byte_size(text)

    result =
      Enum.reduce_while(offsets, [], &normalize_offset(&1, &2, boundaries, text_size))

    case result do
      {:error, _reason} = error -> error
      normalized -> {:ok, Enum.sort(normalized)}
    end
  end

  defp content_offsets(offsets, _text),
    do: {:error, {:invalid_tokenizer_offsets, offsets}}

  defp normalize_offset(
         {start_byte, end_byte},
         normalized,
         boundaries,
         text_size
       )
       when is_integer(start_byte) and is_integer(end_byte) and start_byte >= 0 and
              end_byte <= text_size and start_byte <= end_byte do
    if start_byte == end_byte do
      {:cont, normalized}
    else
      offset =
        {previous_boundary(boundaries, start_byte),
         next_boundary(boundaries, end_byte, text_size)}

      {:cont, [offset | normalized]}
    end
  end

  defp normalize_offset(invalid, _normalized, _boundaries, _text_size),
    do: {:halt, {:error, {:invalid_tokenizer_offset, invalid}}}

  defp grapheme_boundaries("", _offset, boundaries), do: boundaries

  defp grapheme_boundaries(text, offset, boundaries) do
    {grapheme, rest} = String.next_grapheme(text)
    next_offset = offset + byte_size(grapheme)
    grapheme_boundaries(rest, next_offset, MapSet.put(boundaries, next_offset))
  end

  defp previous_boundary(boundaries, offset) do
    if MapSet.member?(boundaries, offset),
      do: offset,
      else: previous_boundary(boundaries, offset - 1)
  end

  defp next_boundary(_boundaries, offset, text_size) when offset >= text_size, do: text_size

  defp next_boundary(boundaries, offset, text_size) do
    if MapSet.member?(boundaries, offset),
      do: offset,
      else: next_boundary(boundaries, offset + 1, text_size)
  end

  defp merge_offset({start_byte, end_byte}, [
         {current_start, current_end, token_count} | rest
       ])
       when start_byte < current_end do
    [{current_start, max(current_end, end_byte), token_count + 1} | rest]
  end

  defp merge_offset({start_byte, end_byte}, units),
    do: [{start_byte, end_byte, 1} | units]

  defp pack_windows(_units, _token_units, total, _size, _step, start, _previous, groups)
       when start >= total,
       do: Enum.reverse(groups)

  defp pack_windows(units, token_units, total, size, step, start, previous, groups) do
    first = unit_index(token_units, start)
    last = unit_index(token_units, min(start + size, total) - 1)
    range = {first, last}

    groups =
      if range == previous do
        groups
      else
        [for(index <- first..last, do: elem(units, index)) | groups]
      end

    pack_windows(units, token_units, total, size, step, start + step, range, groups)
  end

  defp unit_index(nil, token_index), do: token_index
  defp unit_index(token_units, token_index), do: elem(token_units, token_index)
end
