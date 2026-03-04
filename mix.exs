defmodule Chunx.MixProject do
  use Mix.Project

  def project do
    [
      app: :chunx,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      source_url: "https://github.com/preciz/chunx"
    ]
  end

  defp description do
    "An Elixir library for splitting text into meaningful chunks using Token, Word, Sentence, and Semantic strategies."
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/preciz/chunx"}
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
      {:bumblebee, "~> 0.6", only: [:dev, :test]},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:benchee, "~> 1.3", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
