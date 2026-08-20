Mix.install([
  {:chunx, "~> 0.2.0"}
])

defmodule Demo do
  @text """
  Text chunking divides a document into smaller sections. A tokenizer measures
  their size. Different chunkers preserve different boundaries.

  Sentence chunking keeps sentences together. Recursive chunking first tries
  document structure, then falls back to smaller boundaries.
  """

  def run do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")

    show("Token", Chunx.Chunker.Token.chunk(@text, tokenizer, chunk_size: 20))
    show("Word", Chunx.Chunker.Word.chunk(@text, tokenizer, chunk_size: 20))
    show("Sentence", Chunx.Chunker.Sentence.chunk(@text, tokenizer, chunk_size: 20))
    show("Recursive", Chunx.Chunker.Recursive.chunk(@text, tokenizer, chunk_size: 20))
  end

  defp show(name, {:ok, chunks}) do
    IO.puts("\n#{name}")

    Enum.each(chunks, fn chunk ->
      IO.puts("[#{chunk.start_byte}, #{chunk.end_byte}) (#{chunk.token_count} tokens)")
      IO.puts(chunk.text)
    end)
  end
end

Demo.run()
