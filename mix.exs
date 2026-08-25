defmodule Chunx.MixProject do
  use Mix.Project

  @version "0.2.1"

  def project do
    [
      app: :chunx,
      version: @version,
      elixir: "~> 1.17",
      description: description(),
      package: package(),
      docs: docs(),
      deps: deps(),
      source_url: "https://github.com/preciz/chunx",
      test_coverage: [summary: [threshold: 100]]
    ]
  end

  defp description do
    "Text chunking for Elixir with token, word, sentence, recursive, and semantic strategies."
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "LICENSE"],
      source_ref: "v#{@version}"
    ]
  end

  defp package do
    [
      maintainers: ["Barna Kovacs"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/preciz/chunx"}
    ]
  end

  defp deps do
    [
      {:tokenizers, "~> 0.5.1"},
      {:nx, "~> 0.13.1"},
      {:scholar, "~> 0.4.2"},
      {:exla, "~> 0.13.1", only: [:dev, :test]},
      {:bumblebee, "~> 0.7.1", only: [:dev, :test]},
      {:stream_data, "~> 1.4", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    ]
  end
end
