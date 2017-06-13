// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.DefaultLearningRate;
import ch.ethz.idsc.subare.core.td.LearningRate;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;

class Sarsa_Maxbias {
  static void handle(SarsaType type, int n) throws Exception {
    System.out.println(type);
    Maxbias maxbias = new Maxbias(3);
    // final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.7, .1, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = type.supply(maxbias, qsa, learningRate);
    for (int index = 0; index < EPISODES; ++index) {
      // System.out.println(index);
      // Scalar error = DiscreteQsas.distance(qsa, ref);
      // System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(maxbias, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policyInterface);
      for (int count = 0; count < 3; ++count) {
        // System.out.println("" + count);
        ExploringStarts.batch(maxbias, policyInterface, n, sarsa);
      }
    }
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    vs.print(Digits._3);
    System.out.println("---");
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.original, 3);
    handle(SarsaType.expected, 1);
    handle(SarsaType.qlearning, 3);
  }
}
