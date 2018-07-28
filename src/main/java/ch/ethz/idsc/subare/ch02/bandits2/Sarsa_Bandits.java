// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

/** Sarsa applied to bandits */
enum Sarsa_Bandits {
  ;
  static void train(Bandits bandits, SarsaType sarsaType, //
      int batches, Scalar factor, Scalar exponent) throws Exception {
    System.out.println(sarsaType);
    final DiscreteQsa ref = BanditsHelper.getOptimalQsa(bandits); // true q-function, for error measurement
    Tensor epsilon = Subdivide.of(.6, .01, batches);
    DiscreteQsa qsa = DiscreteQsa.build(bandits); // q-function for training, initialized to 0
    // ---
    final Sarsa sarsa = sarsaType.supply(bandits, DefaultLearningRate.of(factor, exponent), qsa);
    // ---
    for (int index = 0; index < batches; ++index) {
      Scalar error1 = Loss.accumulation(bandits, ref, qsa);
      System.out.println(index + " " + epsilon.Get(index).map(Round._2) + " " + error1.map(Round._3));
      Policy policy = EGreedyPolicy.bestEquiprobable(bandits, qsa, epsilon.Get(index));
      // sarsa.supplyPolicy(() -> policy);
      sarsa.setExplore(epsilon.Get(index));
      ExploringStarts.batch(bandits, policy, 1, sarsa);
    }
    System.out.println("---");
    System.out.println("true state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(bandits, ref), Round._3);
    System.out.println("estimated state values:");
    DiscreteUtils.print(DiscreteUtils.createVs(bandits, qsa), Round._3);
  }

  public static void main(String[] args) throws Exception {
    Bandits bandits = new Bandits(20);
    train(bandits, SarsaType.ORIGINAL, 100, RealScalar.of(16), RealScalar.of(1.15));
  }
}
