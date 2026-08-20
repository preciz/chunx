defmodule Chunx.Chunker.SentenceSplitter do
  @moduledoc false

  @spec split(binary(), [binary()]) :: [binary()]
  def split(text, delimiters) when is_binary(text) and is_list(delimiters) do
    if Enum.empty?(delimiters) or
         not Enum.all?(delimiters, &(is_binary(&1) and &1 != "")) do
      raise ArgumentError, "delimiters must contain non-empty strings"
    end

    if text == "", do: [], else: do_split(text, delimiters, 0, [])
  end

  defp do_split(text, _delimiters, offset, acc) when offset == byte_size(text) do
    Enum.reverse(acc)
  end

  defp do_split(text, delimiters, offset, acc) do
    case next_delimiter(text, delimiters, offset) do
      nil ->
        part = binary_part(text, offset, byte_size(text) - offset)
        Enum.reverse([part | acc])

      {position, length} ->
        end_byte = position + length
        part = binary_part(text, offset, end_byte - offset)
        do_split(text, delimiters, end_byte, [part | acc])
    end
  end

  defp next_delimiter(text, delimiters, offset) do
    scope = {offset, byte_size(text) - offset}

    delimiters
    |> Enum.flat_map(fn delimiter ->
      case :binary.match(text, delimiter, scope: scope) do
        {position, length} -> [{position, length}]
        :nomatch -> []
      end
    end)
    |> Enum.min_by(fn {position, length} -> {position, -length} end, fn -> nil end)
  end
end
