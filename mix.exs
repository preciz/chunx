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
      {:nx, "~> 0.9.2"},
      {:scholar, "~> 0.4.0"},
      {:exla, "~> 0.9.2", only: [:dev, :test]},
      {:bumblebee, "~> 0.6.0", only: [:dev, :test]}
    ]
  end
end
