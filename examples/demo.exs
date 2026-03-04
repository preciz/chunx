Mix.install([
  {:chunx, "~> 0.1.0"}
])

defmodule Demo do
  def run do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("gpt2")
    
    text = """
    The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements.
    
    # Heading 1
    This is a paragraph with some **bold text** and _italic text_.
    
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    """

    IO.puts("=== Token-based Chunking ===")
    {:ok, token_chunks} = Chunx.Chunker.Token.chunk(text, tokenizer, chunk_size: 50, chunk_overlap: 10)
    Enum.each(Enum.with_index(token_chunks), fn {chunk, i} ->
      IO.puts("Chunk #{i} (Tokens: #{chunk.token_count}):\n#{chunk.text}\n")
    end)

    IO.puts("\n=== Word-based Chunking ===")
    {:ok, word_chunks} = Chunx.Chunker.Word.chunk(text, tokenizer, chunk_size: 50, chunk_overlap: 10)
    Enum.each(Enum.with_index(word_chunks), fn {chunk, i} ->
      IO.puts("Chunk #{i} (Tokens: #{chunk.token_count}):\n#{chunk.text}\n")
    end)

    IO.puts("\n=== Sentence-based Chunking ===")
    {:ok, sentence_chunks} = Chunx.Chunker.Sentence.chunk(text, tokenizer, chunk_size: 50, chunk_overlap: 10)
    Enum.each(Enum.with_index(sentence_chunks), fn {chunk, i} ->
      IO.puts("Chunk #{i} (Tokens: #{chunk.token_count}):\n#{chunk.text}\n")
    end)
  end
end

Demo.run()
