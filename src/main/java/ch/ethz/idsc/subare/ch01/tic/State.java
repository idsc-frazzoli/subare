// code by jph
// adapted from code by Shangtong Zhang
package ch.ethz.idsc.subare.ch01.tic;

import java.io.Serializable;
import java.util.Arrays;

class State implements Serializable {
  public static String[] sym = { "0", "x", "*" };
  public static State empty = new State(new int[9]);

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
  public boolean equals(Object obj) {
    return hashCode == ((State) obj).hashCode;
  }

  boolean isEnd() {
    return winner != null;
  }

  State nextState(int pos, int symbol) {
    int[] copy = Arrays.copyOf(data, 9);
    copy[pos] = symbol;
    State newState = new State(copy);
    return newState;
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
    StringBuilder myStringBuilder = new StringBuilder();
    for (int del = 0; del < 3; ++del) {
      for (int ofs = 0; ofs < 3; ++ofs) {
        myStringBuilder.append(sym[hashing(data[3 * del + ofs])]);
        myStringBuilder.append(" ");
      }
      myStringBuilder.append(" | ");
      if (del == 0) {
        myStringBuilder.append("hash=" + hashCode());
      }
      if (del < 2) {
        myStringBuilder.append('\n');
      }
    }
    if (winner != null)
      myStringBuilder.append("winner=" + winner + "\n");
    return myStringBuilder.toString();
  }
}
