defmodule Chunx.Helper do
  @moduledoc false

  @spec validate_text(binary()) :: :ok | {:error, {:invalid_text, :invalid_utf8}}
  def validate_text(text) when is_binary(text) do
    if String.valid?(text), do: :ok, else: {:error, {:invalid_text, :invalid_utf8}}
  end

  @spec median(nonempty_list(number())) :: number()
  def median(values) do
    sorted = Enum.sort(values)
    len = length(sorted)
    mid = div(len, 2)

    case rem(len, 2) do
      0 -> (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
      1 -> Enum.at(sorted, mid)
    end
  end

  @spec standard_deviation(nonempty_list(number())) :: float()
  def standard_deviation(values) do
    count = length(values)
    mean = Enum.sum(values) / count

    variance =
      Enum.reduce(values, 0, fn value, sum ->
        sum + :math.pow(value - mean, 2)
      end) / count

    :math.sqrt(variance)
  end
end
