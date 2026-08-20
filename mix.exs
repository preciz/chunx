defmodule Chunx.MixProject do
  use Mix.Project

  @version "0.2.0"

  def project do
    [
      app: :chunx,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      source_url: "https://github.com/preciz/chunx"
    ]
  end

  defp description do
    "An Elixir library for splitting text into meaningful chunks using Token, Word, Sentence, Recursive, and Semantic strategies."
  end

  defp package do
    [
      maintainers: ["Barna Kovacs"],
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
      {:nx, "~> 0.13.1"},
      {:scholar, "~> 0.4.2"},
      {:exla, "~> 0.13.1", only: [:dev, :test]},
      {:bumblebee, "~> 0.7.1", only: [:dev, :test]},
      {:stream_data, "~> 1.4", only: [:dev, :test]},
      {:benchee, "~> 1.3", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    ]
  end
end
