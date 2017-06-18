// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

/** Double Sarsa for maximization bias */
class Double_Bandits {
  static void handle(SarsaType sarsaType, int n) throws Exception {
    System.out.println("double " + sarsaType);
    Bandits bandits = new Bandits(20);
    final DiscreteQsa ref = BanditsHelper.getOptimalQsa(bandits);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.3, .01, EPISODES); // used in egreedy
    DiscreteQsa qsa1 = DiscreteQsa.build(bandits);
    DiscreteQsa qsa2 = DiscreteQsa.build(bandits);
    DoubleSarsa doubleSarsa = new DoubleSarsa(sarsaType, bandits, //
        qsa1, qsa2, //
        DefaultLearningRate.of(15, 1.31), //
        DefaultLearningRate.of(15, 1.31));
    for (int index = 0; index < EPISODES; ++index) {
      Scalar explore = epsilon.Get(index);
      Scalar error = Loss.accumulation(bandits, ref, qsa1);
      if (EPISODES - 10 < index)
        System.out.println(index + " " + explore.map(Round._2) + " " + error.map(Round._3));
      Policy policy1 = EGreedyPolicy.bestEquiprobable(bandits, qsa1, explore);
      Policy policy2 = EGreedyPolicy.bestEquiprobable(bandits, qsa2, explore);
      doubleSarsa.setPolicy(policy1, policy2);
      ExploringStarts.batch(bandits, doubleSarsa.getEGreedy(explore), n, doubleSarsa);
    }
    System.out.println("---");
    System.out.println("true state values:");
    DiscreteUtils.createVs(bandits, ref).print(Round._3);
    System.out.println("estimated state values:");
    DiscreteUtils.createVs(bandits, qsa1).print(Round._3);
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.qlearning, 1);
    handle(SarsaType.expected, 3);
    handle(SarsaType.qlearning, 2);
  }
}
