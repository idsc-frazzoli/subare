// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Julian's idea
 * Prisoners Dilemma */
class Training {
  @SuppressWarnings("unused")
  static void train(Agent a1, Agent a2) {
    int epochs = 2000;
    Tensor r2 = Tensors.matrixDouble(new double[][] { //
        { 1, 4 }, //
        { 0, 3 } });
    // { 3, 0 }, //
    // { 4, 1 } });
    Tensor rn = Tensors.matrixDouble(new double[][] { //
        { 1, -2 }, //
        { 2, -1 } });
    Judger judger = new Judger(r2, a1, a2);
    // ---
    for (int round = 0; round < epochs; ++round)
      judger.play();
    judger.ranking();
  }

  public static void main(String[] args) {
    final int size = AgentSupplier.values().length;
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
        System.out.println(i1 + "---" + i2);
        train( //
            AgentSupplier.values()[i1].supplier.get(), //
            AgentSupplier.values()[i2].supplier.get());
      }
    }
  }
}
