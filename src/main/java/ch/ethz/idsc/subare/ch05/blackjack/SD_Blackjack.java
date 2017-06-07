package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

public class SD_Blackjack {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  static void handle(SarsaType type) throws Exception {
    System.out.println(type);
    final Blackjack blackjack = new Blackjack();
    int EPISODES = 40;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/blackjack_qsa_" + type + ".gif"), 200);
    for (int index = 0; index < EPISODES; ++index) {
      // Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + epsilon.Get(index).map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(blackjack, qsa, epsilon.Get(index));
      Sarsa sarsa = type.supply(blackjack, qsa, RealScalar.of(.2), policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStartsBatch.apply(blackjack, sarsa, policyInterface);
      gsw.append(ImageFormat.of(BlackjackHelper.joinAll(blackjack, qsa)));
    }
    gsw.close();
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.qlearning);
  }
}
