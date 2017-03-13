// code by jph
package ch.ethz.idsc.subare.ch02.prison;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Mean;

/** Julian's idea
 * Prisoners Dilemma */
class Training {
  @SuppressWarnings("unused")
  static Tensor train(Agent a1, Agent a2) {
    int epochs = 1000;
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
    return judger.ranking();
  }

  public static void main(String[] args) {
    final int size = AgentSupplier.values().length;
    for (int i1 = 0; i1 < size; ++i1) {
      for (int i2 = i1; i2 < size; ++i2) {
         
        Tensor table = Tensors.empty();
        for (int run = 0; run < 100; ++run) {
          Agent a1 = AgentSupplier.values()[i1].supplier.get();
          Agent a2 = AgentSupplier.values()[i2].supplier.get();
          table.append(train(a1, a2));
        }
        {
          Agent a1 = AgentSupplier.values()[i1].supplier.get();
          Agent a2 = AgentSupplier.values()[i2].supplier.get();
          Tensor mean = Mean.of(table);
          RealScalar g1 = (RealScalar) mean.Get(0);
          RealScalar g2 = (RealScalar) mean.Get(1);
          System.out.println(i1 + "---" + i2);
          System.out.println(String.format("%s %6.3f", a1.getAbsDesc(), g1.getRealDouble()));
          System.out.println(String.format("%s %6.3f", a2.getAbsDesc(), g2.getRealDouble()));
        }
      }
    }
  }
}
