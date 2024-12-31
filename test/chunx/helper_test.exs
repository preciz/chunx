defmodule Chunx.HelperTest do
  use ExUnit.Case, async: true
  alias Chunx.Helper

  describe "median/1" do
    test "returns correct median for odd length list" do
      assert Helper.median([1, 2, 3]) == 2
      # unsorted input
      assert Helper.median([1, 3, 2]) == 2
      assert Helper.median([1, 5, 9]) == 5
    end

    test "returns correct median for even length list" do
      assert Helper.median([1, 2, 3, 4]) == 2.5
      # unsorted input
      assert Helper.median([1, 4, 2, 3]) == 2.5
      assert Helper.median([1, 2]) == 1.5
    end

    test "handles single element list" do
      assert Helper.median([42]) == 42
    end

    test "handles decimal numbers" do
      assert Helper.median([1.5, 2.5, 3.5]) == 2.5
    end
  end

  describe "standard_deviation/1" do
    test "returns correct standard deviation for basic cases" do
      # Known values: [2, 4, 4, 4, 5, 5, 7, 9] has std dev of 2.0
      assert_in_delta Helper.standard_deviation([2, 4, 4, 4, 5, 5, 7, 9]), 2.0, 0.0001

      # [1, 2, 3] has std dev of ~0.816
      assert_in_delta Helper.standard_deviation([1, 2, 3]), 0.816496580927726, 0.0001
    end

    test "returns 0 for identical values" do
      assert Helper.standard_deviation([5, 5, 5]) == 0.0
      assert Helper.standard_deviation([42, 42]) == 0.0
    end

    test "handles single element list" do
      assert Helper.standard_deviation([42]) == 0.0
    end

    test "handles decimal numbers" do
      assert_in_delta Helper.standard_deviation([1.5, 2.0, 2.5]), 0.4082482904638631, 0.0001
    end
  end
end
