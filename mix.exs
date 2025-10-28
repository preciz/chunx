defmodule Chunx.MixProject do
  use Mix.Project

  def project do
    [
      app: :chunx,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:tokenizers, "~> 0.5.1"},
      {:nx, "~> 0.10"},
      {:scholar, "~> 0.4"},
      {:exla, "~> 0.10", only: [:dev, :test]},
      {:bumblebee, "~> 0.6.3", only: [:dev, :test]}
    ]
  end
end
