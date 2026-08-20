defmodule Chunx.Helper do
  @moduledoc false

  @doc """
  Provides math and statistic helper functions for array analysis.
  """
  @spec median([number()]) :: number()
  def median(values) do
    sorted = Enum.sort(values)
    len = length(sorted)
    mid = div(len, 2)

    case rem(len, 2) do
      0 -> (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
      1 -> Enum.at(sorted, mid)
    end
  end

  @spec standard_deviation([number()]) :: float()
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
