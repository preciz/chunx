embedding_integration? =
  System.get_env("CHUNX_EMBEDDING_INTEGRATION", "")
  |> String.downcase()
  |> then(&(&1 in ["1", "true", "yes"]))

exclude = if embedding_integration?, do: [], else: [embedding_integration: true]

ExUnit.start(exclude: exclude)
