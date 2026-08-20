defmodule Chunx.EmbeddingIntegrationTest do
  use ExUnit.Case, async: false

  alias Chunx.Chunker.Semantic
  alias Chunx.Chunker.Semantic.Sentences
  alias Scholar.Metrics.Distance

  @moduletag :integration
  @moduletag timeout: 600_000

  @model "sentence-transformers/all-MiniLM-L6-v2"

  setup_all do
    repository = {:hf, @model}

    {:ok, model_info} = Bumblebee.load_model(repository)
    {:ok, bumblebee_tokenizer} = Bumblebee.load_tokenizer(repository)
    {:ok, chunk_tokenizer} = Tokenizers.Tokenizer.from_pretrained(@model)

    serving =
      Bumblebee.Text.text_embedding(model_info, bumblebee_tokenizer,
        output_attribute: :hidden_state,
        output_pool: :mean_pooling,
        embedding_processor: :l2_norm,
        compile: [batch_size: 8, sequence_length: 64],
        defn_options: [compiler: EXLA]
      )

    %{embedding_fun: embedding_fun(serving), tokenizer: chunk_tokenizer}
  end

  test "real embeddings are aligned, normalized, finite, and semantically meaningful", %{
    embedding_fun: embedding_fun
  } do
    texts = [
      "A cat is resting on a warm rug.",
      "A feline lies comfortably on a carpet.",
      "Database indexes speed up selective queries."
    ]

    embeddings = embedding_fun.(texts)

    assert length(embeddings) == length(texts)
    assert Enum.uniq_by(embeddings, &Nx.shape/1) |> length() == 1

    Enum.each(embeddings, fn embedding ->
      assert {dimension} = Nx.shape(embedding)
      assert dimension > 1
      assert Enum.all?(Nx.to_flat_list(embedding), &finite_number?/1)
      assert_in_delta vector_norm(embedding), 1.0, 1.0e-4
    end)

    [cat, feline, database] = embeddings
    assert cosine_similarity(cat, cat) > 0.999
    assert cosine_similarity(cat, feline) > cosine_similarity(cat, database)
  end

  test "sentence preparation sends the exact groups to the model and preserves tensor alignment",
       %{
         embedding_fun: embedding_fun,
         tokenizer: tokenizer
       } do
    text =
      "Cats sleep on soft blankets. Felines enjoy warm resting places. " <>
        "Database indexes accelerate queries."

    sentences = Sentences.split_sentences(text, ["."], 0)
    expected_groups = Sentences.build_sentence_groups(sentences, 1)
    expected_embeddings = embedding_fun.(expected_groups)

    prepared =
      Sentences.prepare_sentences(text, tokenizer, embedding_fun,
        delimiters: ["."],
        min_chars_per_sentence: 0,
        similarity_window: 1
      )

    assert length(prepared) == length(expected_groups)

    Enum.zip(prepared, expected_embeddings)
    |> Enum.each(fn {sentence, expected_embedding} ->
      assert %Nx.Tensor{} = sentence.embedding
      assert Nx.shape(sentence.embedding) == Nx.shape(expected_embedding)
      assert_in_delta cosine_similarity(sentence.embedding, expected_embedding), 1.0, 1.0e-5
    end)
  end

  test "real semantic embeddings split a topic transition end to end", %{
    embedding_fun: embedding_fun,
    tokenizer: tokenizer
  } do
    first = "Cats often sleep on warm blankets."
    second = "A relaxed feline likes resting on a soft rug."
    third = "Database indexes make selective queries faster."
    fourth = "A query planner can choose an efficient database index."
    sentences = [first, " " <> second, " " <> third, " " <> fourth]
    text = Enum.join(sentences)

    [cat, feline, database, indexed_database] = embedding_fun.(sentences)
    cat_similarity = cosine_similarity(cat, feline)
    transition_similarity = cosine_similarity(feline, database)
    database_similarity = cosine_similarity(database, indexed_database)

    assert cat_similarity > transition_similarity
    assert database_similarity > transition_similarity

    threshold = (cat_similarity + transition_similarity) / 2 + 1.0e-6

    assert {:ok, [cat_chunk, database_chunk]} =
             Semantic.chunk(text, tokenizer, embedding_fun,
               threshold: threshold,
               min_sentences: 2,
               chunk_size: 512,
               delimiters: ["."],
               min_chars_per_sentence: 0,
               similarity_window: 0
             )

    assert cat_chunk.text == first <> " " <> second
    assert database_chunk.text == " " <> third <> " " <> fourth
    assert Enum.map_join([cat_chunk, database_chunk], & &1.text) == text

    assert Enum.all?(
             cat_chunk.sentences ++ database_chunk.sentences,
             &match?(%Nx.Tensor{}, &1.embedding)
           )
  end

  defp embedding_fun(serving) do
    fn texts ->
      serving
      |> Nx.Serving.run(texts)
      |> Enum.map(& &1.embedding)
    end
  end

  defp cosine_similarity(left, right),
    do: 1.0 - Nx.to_number(Distance.cosine(left, right))

  defp vector_norm(tensor) do
    tensor
    |> Nx.multiply(tensor)
    |> Nx.sum()
    |> Nx.sqrt()
    |> Nx.to_number()
  end

  defp finite_number?(number),
    do: is_number(number) and abs(number) < 1.0e308
end
