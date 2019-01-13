// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.sca.Round;

// FIXME this demo throws an exception
enum Sarsa_Blackjack {
  ;
  static void handle(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    final Blackjack blackjack = new Blackjack();
    int batches = 40;
    Tensor epsilon = Subdivide.of(.1, .01, batches); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    try (AnimationWriter animationWriter = AnimationWriter.of(HomeDirectory.Pictures("blackjack_qsa_" + sarsaType + ".gif"), 200)) {
      Sarsa sarsa = sarsaType.sarsa(blackjack, DefaultLearningRate.of(2, 0.6), qsa, sac, policy);
      for (int index = 0; index < batches; ++index) {
        // Scalar error = DiscreteQsas.distance(qsa, ref);
        System.out.println(index + " " + epsilon.Get(index).map(Round._2));
        // sarsa.supplyPolicy(() -> policy);
        for (int count = 0; count < 10; ++count)
          ExploringStarts.batch(blackjack, policy, sarsa);
        animationWriter.append(BlackjackHelper.joinAll(blackjack, qsa));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.QLEARNING);
  }
}
