defmodule Chunx.Helper do
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
    mean = Enum.sum(values) / length(values)

    variance =
      values
      |> Enum.map(fn x -> :math.pow(x - mean, 2) end)
      |> Enum.sum()
      |> Kernel./(length(values))

    :math.sqrt(variance)
  end
end
