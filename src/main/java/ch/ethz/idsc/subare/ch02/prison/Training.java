// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Julian's idea: Prisoners' Dilemma */
class Training {
  /** rewards average at 2 */
  static final Tensor r2 = Tensors.matrixInt(new int[][] { //
      { 1, 4 }, //
      { 0, 3 } });

  /** @param a1
   * @param a2
   * @param epochs
   * @return tensor of rewards averaged over number of actions */
  static Tensor train(Agent a1, Agent a2, int epochs) {
    Judger judger = new Judger(r2, a1, a2);
    for (int round = 0; round < epochs; ++round)
      judger.play();
    return judger.ranking();
  }
}
