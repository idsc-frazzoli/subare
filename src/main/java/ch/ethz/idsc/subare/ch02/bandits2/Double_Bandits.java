// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

/** Double Sarsa for maximization bias */
enum Double_Bandits {
  ;
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("double " + sarsaType);
    Bandits bandits = new Bandits(20);
    final DiscreteQsa ref = BanditsHelper.getOptimalQsa(bandits);
    int batches = 100;
    Tensor epsilon = Subdivide.of(.3, .01, batches); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(bandits);
    DiscreteQsa qsa2 = DiscreteQsa.build(bandits);
    DoubleSarsa doubleSarsa = new DoubleSarsa(sarsaType, bandits, //
        qsa1, qsa2, //
        DefaultLearningRate.of(15, 1.31), //
        DefaultLearningRate.of(15, 1.31));
    for (int index = 0; index < batches; ++index) {
      Scalar error = Loss.accumulation(bandits, ref, qsa1);
      if (batches - 10 < index)
        System.out.println(index + " " + epsilon.Get(index).map(Round._2) + " " + error.map(Round._3));
      doubleSarsa.setExplore(epsilon.Get(index));
      ExploringStarts.batch(bandits, doubleSarsa.getEGreedy(), n, doubleSarsa);
    }
    System.out.println("---");
    System.out.println("true state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(bandits, ref), Round._3);
    System.out.println("estimated state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(bandits, qsa1), Round._3);
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING, 1);
    handle(SarsaType.EXPECTED, 3);
    handle(SarsaType.QLEARNING, 2);
  }
}
