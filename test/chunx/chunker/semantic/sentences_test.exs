defmodule Chunx.Chunker.Semantic.SentencesTest do
  use ExUnit.Case, async: true

  alias Chunx.Chunker.Semantic.Sentences
  alias Chunx.Chunk

  @titanic """
  Titanic was a ship that sank in the North Atlantic Ocean in 1912 after hitting an iceberg.
  The ship was on its maiden voyage from Southampton to New York City.
  There were over 2,000 passengers and crew on board, and more than 1,500 of them died.
  Mr. Smith was the captain of the Titanic.
  He was an experienced sailor and had been with the White Star Line for many years.
  He was known for his calm demeanor and quick thinking in emergencies.
  """

  @complex_markdown """
  # Heading 1
  This is a paragraph with some **bold text** and _italic text_.
  ## Heading 2
  - Bullet point 1
  - Bullet point 2 with `inline code`
  ```python
  # Code block
  def hello_world():
      print("Hello, world!")
  ```
  Another paragraph with [a link](https://example.com) and an image:
  ![Alt text](https://example.com/image.jpg)
  > A blockquote with multiple lines
  > that spans more than one line.
  Finally, a paragraph at the end.
  """

  setup do
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("minishlab/potion-base-8M")

    %{
      tokenizer: tokenizer,
      embedding_fun: fn inputs ->
        Enum.map(inputs, fn _ -> Nx.tensor(List.duplicate(1.0, 768)) end)
      end
    }
  end

  describe "prepare_sentences/4" do
    test "behaves exactly like python implementation 1", %{
      tokenizer: tokenizer,
      embedding_fun: embedding_fun
    } do
      sentences =
        Sentences.prepare_sentences(
          @titanic,
          tokenizer,
          embedding_fun,
          similarity_window: 1,
          min_chars_per_sentence: 12
        )

      assert [
               %Chunk{
                 text:
                   "Titanic was a ship that sank in the North Atlantic Ocean in 1912 after hitting an iceberg.\n"
               },
               %Chunk{
                 text: "The ship was on its maiden voyage from Southampton to New York City.\n"
               },
               %Chunk{
                 text:
                   "There were over 2,000 passengers and crew on board, and more than 1,500 of them died.\nMr."
               },
               %Chunk{text: " Smith was the captain of the Titanic.\n"},
               %Chunk{
                 text:
                   "He was an experienced sailor and had been with the White Star Line for many years.\n"
               },
               %Chunk{
                 text: "He was known for his calm demeanor and quick thinking in emergencies.\n"
               }
             ] = sentences
    end

    test "behaves exactly like python implementation 2", %{
      tokenizer: tokenizer,
      embedding_fun: embedding_fun
    } do
      sentences =
        Sentences.prepare_sentences(
          @titanic,
          tokenizer,
          embedding_fun,
          similarity_window: 1,
          min_chars_per_sentence: 1
        )

      assert [
               %Chunk{
                 text:
                   "Titanic was a ship that sank in the North Atlantic Ocean in 1912 after hitting an iceberg.\n"
               },
               %Chunk{
                 text: "The ship was on its maiden voyage from Southampton to New York City.\n"
               },
               %Chunk{
                 text:
                   "There were over 2,000 passengers and crew on board, and more than 1,500 of them died.\n"
               },
               %Chunk{text: "Mr."},
               %Chunk{text: " Smith was the captain of the Titanic.\n"},
               %Chunk{
                 text:
                   "He was an experienced sailor and had been with the White Star Line for many years.\n"
               },
               %Chunk{
                 text: "He was known for his calm demeanor and quick thinking in emergencies.\n"
               }
             ] = sentences
    end

    test "behaves exactly like python implementation 3", %{
      tokenizer: tokenizer,
      embedding_fun: embedding_fun
    } do
      sentences =
        Sentences.prepare_sentences(
          @titanic,
          tokenizer,
          embedding_fun,
          delimiters: ["C"]
        )

      assert [
               %Chunx.Chunk{
                 text:
                   "Titanic was a ship that sank in the North Atlantic Ocean in 1912 after hitting an iceberg.\nThe ship was on its maiden voyage from Southampton to New York C"
               },
               %Chunx.Chunk{
                 text:
                   "ity.\nThere were over 2,000 passengers and crew on board, and more than 1,500 of them died.\nMr. Smith was the captain of the Titanic.\nHe was an experienced sailor and had been with the White Star Line for many years.\nHe was known for his calm demeanor and quick thinking in emergencies.\n"
               }
             ] = sentences
    end

    test "maintains correct text indices", %{tokenizer: tokenizer, embedding_fun: embedding_fun} do
      sentences =
        Sentences.prepare_sentences(
          @complex_markdown,
          tokenizer,
          embedding_fun
        )

      Enum.each(sentences, fn chunk ->
        extracted_text =
          String.slice(@complex_markdown, chunk.start_byte, chunk.end_byte - chunk.start_byte)

        assert chunk.text == extracted_text
      end)
    end
  end

  describe "build_sentence_groups/2" do
    test "returns original sentences when window is 0" do
      sentences = ["First.", "Second.", "Third."]
      assert Sentences.build_sentence_groups(sentences, 0) == sentences
    end

    test "builds groups with window size 1" do
      sentences = ["First.", "Second.", "Third.", "Fourth."]
      result = Sentences.build_sentence_groups(sentences, 1)

      assert result == [
               # First sentence + 1 after
               "First.Second.",
               # Second sentence + 1 before/after
               "First.Second.Third.",
               # Third sentence + 1 before/after
               "Second.Third.Fourth.",
               # Fourth sentence + 1 before
               "Third.Fourth."
             ]
    end

    test "builds groups with window size 2" do
      sentences = ["One.", "Two.", "Three.", "Four.", "Five."]
      result = Sentences.build_sentence_groups(sentences, 2)

      assert result == [
               # First + 2 after
               "One.Two.Three.",
               # Second + 2 before/after
               "One.Two.Three.Four.",
               # Middle with 2 on each side
               "One.Two.Three.Four.Five.",
               # Fourth + 2 before/after
               "Two.Three.Four.Five.",
               # Last + 2 before
               "Three.Four.Five."
             ]
    end

    test "handles empty list" do
      assert Sentences.build_sentence_groups([], 1) == []
    end

    test "handles single sentence" do
      assert Sentences.build_sentence_groups(["Only."], 1) == ["Only."]
    end
  end

  describe "combine_short_sentences/2" do
    test "combines sentences shorter than min_chars" do
      splits = ["Hi", " there", " friend", "! This is a longer sentence."]
      result = Sentences.combine_short_sentences(splits, 12)
      assert result == ["Hi there friend", "! This is a longer sentence."]
    end

    test "keeps sentences longer than min_chars separate" do
      splits = ["Hello world!", " Another greeting!", " Hi!"]
      result = Sentences.combine_short_sentences(splits, 10)
      assert result == ["Hello world!", " Another greeting! Hi!"]
    end

    test "handles empty list" do
      assert Sentences.combine_short_sentences([], 10) == []
    end

    test "handles single sentence" do
      assert Sentences.combine_short_sentences(["Hello!"], 10) == ["Hello!"]
    end

    test "combines multiple short sentences at the end" do
      splits = ["Long sentence here.", " Hi", " my", " friend"]
      result = Sentences.combine_short_sentences(splits, 10)
      assert result == ["Long sentence here. Hi my friend"]
    end

    test "respects whitespace in length calculations" do
      splits = ["  Hi  ", "  there  ", "  friend  "]
      result = Sentences.combine_short_sentences(splits, 15)
      assert result == ["  Hi    there    friend  "]
    end
  end

  describe "split_sentences/4" do
    test "splits text on basic delimiters" do
      text = "Hello. How are you? I'm good! Thanks."
      result = Sentences.split_sentences(text, "🦛", [".", "!", "?"], 1)
      assert result == ["Hello.", " How are you?", " I'm good!", " Thanks."]
    end

    test "handles newlines as delimiters" do
      text = "Line one.\nLine two\nLine three."
      result = Sentences.split_sentences(text, "🦛", [".", "\n"], 1)
      assert result == ["Line one.\n", "Line two\n", "Line three."]
    end

    test "handles custom delimiters" do
      text = "Part 1;Part 2;Part 3"
      result = Sentences.split_sentences(text, "🦛", [";"], 1)
      assert result == ["Part 1;", "Part 2;", "Part 3"]
    end

    test "handles empty text" do
      assert Sentences.split_sentences("", "🦛", [".", "!", "?"], 1) == []
    end

    test "handles text with no delimiters" do
      text = "This is a single sentence without delimiters"
      result = Sentences.split_sentences(text, "🦛", [".", "!", "?"], 1)
      assert result == ["This is a single sentence without delimiters"]
    end

    test "handles multiple consecutive delimiters" do
      text = "Hello...World!!!"
      result = Sentences.split_sentences(text, "🦛", [".", "!"], 1)
      assert result == ["Hello.", ".", ".", "World!", "!", "!"]
    end
  end

  describe "find_sentence_indices/2" do
    test "finds correct indices in simple text" do
      text = "First. Second. Third."
      sentences = ["First.", " Second.", " Third."]

      result = Sentences.find_sentence_indices(text, sentences)

      assert result == [
               {"First.", 0, 6},
               {" Second.", 6, 14},
               {" Third.", 14, 21}
             ]
    end

    test "handles sentences with newlines" do
      text = "Line one.\nLine two.\nLine three."
      sentences = ["Line one.", "\nLine two.", "\nLine three."]

      result = Sentences.find_sentence_indices(text, sentences)

      assert result == [
               {"Line one.", 0, 9},
               {"\nLine two.", 9, 19},
               {"\nLine three.", 19, 31}
             ]
    end

    test "handles empty sentences" do
      text = "Hello..World"
      sentences = ["Hello", ".", ".", "World"]

      result = Sentences.find_sentence_indices(text, sentences)

      assert result == [
               {"Hello", 0, 5},
               {".", 5, 6},
               {".", 6, 7},
               {"World", 7, 12}
             ]
    end

    test "handles unicode characters" do
      text = "Hello 👋. World 🌍!"
      sentences = ["Hello 👋.", " World 🌍!"]

      result = Sentences.find_sentence_indices(text, sentences)

      assert result == [
               {"Hello 👋.", 0, 11},
               {" World 🌍!", 11, 23}
             ]
    end

    test "handles overlapping substrings" do
      text = "The cat sat on the mat. The cat ran away."
      sentences = ["The cat sat on the mat.", " The cat ran away."]

      result = Sentences.find_sentence_indices(text, sentences)

      assert result == [
               {"The cat sat on the mat.", 0, 23},
               {" The cat ran away.", 23, 41}
             ]
    end
  end
end
