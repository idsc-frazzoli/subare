// code by jph
// adapted from code by Shangtong Zhang
package ch.ethz.idsc.subare.ch01.tic;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/* package */ class State implements Serializable {
  public static final String[] SYMBOL = { "0", "x", "*" };
  public static final State EMPTY = new State(new int[9]);

  public static int hashing(int value) {
    return value == -1 ? 2 : value;
  }

  // the board is represented by a n * n array,
  // +1 represents chessman of the player who moves first,
  // -1 represents chessman of another player
  // 0 represents empty position
  final int[] data;
  final int hashCode;
  // 1: beginner won
  // -1: 2nd won
  // 0: draw
  // null: game unfinished
  final Integer winner;

  State(int[] data) {
    this.data = data;
    hashCode = computeHash();
    winner = new WinnerStatus(data).winner;
  }

  private int computeHash() {
    int hash = 0;
    for (int value : data) {
      hash *= 3;
      hash += hashing(value);
    }
    return hash;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public boolean equals(Object object) {
    return object instanceof State //
        ? hashCode == ((State) object).hashCode
        : false;
  }

  boolean isEnd() {
    return Objects.nonNull(winner);
  }

  State nextState(int pos, int symbol) {
    int[] copy = Arrays.copyOf(data, 9);
    copy[pos] = symbol;
    return new State(copy);
  }

  public static final int[] indexrotated = { 2, 5, 8, 1, 4, 7, 0, 3, 6 };
  public static final int[] indexmirror = { 2, 1, 0, 5, 4, 3, 8, 7, 6 };

  State getRotated() {
    int[] copy = new int[9];
    for (int pos = 0; pos < 9; ++pos)
      copy[pos] = data[indexrotated[pos]];
    return new State(copy);
  }

  State getMirrored() {
    int[] copy = new int[9];
    for (int pos = 0; pos < 9; ++pos)
      copy[pos] = data[indexmirror[pos]];
    return new State(copy);
  }

  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    for (int del = 0; del < 3; ++del) {
      for (int ofs = 0; ofs < 3; ++ofs) {
        stringBuilder.append(SYMBOL[hashing(data[3 * del + ofs])]);
        stringBuilder.append(" ");
      }
      stringBuilder.append(" | ");
      if (del == 0) {
        stringBuilder.append("hash=" + hashCode());
      }
      if (del < 2) {
        stringBuilder.append('\n');
      }
    }
    if (isEnd())
      stringBuilder.append("winner=" + winner + "\n");
    return stringBuilder.toString();
  }
}
