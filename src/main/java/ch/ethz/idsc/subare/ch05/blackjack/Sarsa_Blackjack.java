// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

public class Sarsa_Blackjack {
  static void handle(SarsaType type) throws Exception {
    System.out.println(type);
    final Blackjack blackjack = new Blackjack();
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("blackjack_qsa_" + type + ".gif"), 200);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = type.supply(blackjack, qsa, learningRate);
    for (int index = 0; index < EPISODES; ++index) {
      // Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + epsilon.Get(index).map(Digits._2));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(blackjack, qsa, epsilon.Get(index));
      sarsa.setPolicyInterface(policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStarts.batch(blackjack, policyInterface, sarsa);
      gsw.append(ImageFormat.of(BlackjackHelper.joinAll(blackjack, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.qlearning);
  }
}
