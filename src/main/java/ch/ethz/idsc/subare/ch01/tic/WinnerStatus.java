// code by jph
package ch.ethz.idsc.subare.ch01.tic;

import java.util.Arrays;

class WinnerStatus {
  final int data[];
  private Integer private_winner = null;
  final Integer winner;

  WinnerStatus(int data[]) {
    this.data = data;
    winner = computeWinner();
  }

  private void computeWinner(int ofs, int del) {
    int sum = data[ofs] + data[ofs + del] + data[ofs + del + del];
    if (sum == 3)
      private_winner = 1;
    else //
    if (sum == -3)
      private_winner = -1;
  }

  private Integer computeWinner() {
    int sum = Arrays.stream(data).map(Math::abs).sum();
    if (sum == 9)
      private_winner = 0;
    // ---
    for (int ofs = 0; ofs < 3; ++ofs) {
      computeWinner(3 * ofs, 1);
      computeWinner(1 * ofs, 3);
    }
    // --- diagonals
    computeWinner(0, 4);
    computeWinner(2, 2);
    // ---
    return private_winner;
  }
}
