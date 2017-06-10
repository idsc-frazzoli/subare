// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.sca.Round;

class Sarsa_Maxbias {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.001));

  static void handle(SarsaType type, int n) throws Exception {
    System.out.println(type);
    Maxbias maxbias = new Maxbias(3);
    // final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int EPISODES = 100;
    Tensor epsilon = Subdivide.of(.7, .1, EPISODES);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    for (int index = 0; index < EPISODES; ++index) {
      // System.out.println(index);
      // Scalar error = DiscreteQsas.distance(qsa, ref);
      // System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(maxbias, qsa, epsilon.Get(index));
      Sarsa sarsa = type.supply(maxbias, qsa, RealScalar.of(.2), policyInterface);
      for (int count = 0; count < 3; ++count) {
        // System.out.println("" + count);
        ExploringStarts.batch(maxbias, policyInterface, n, sarsa);
      }
    }
    DiscreteVs vs = DiscreteUtils.createVs(maxbias, qsa);
    vs.print(ROUND);
  }

  public static void main(String[] args) throws Exception {
    handle(SarsaType.original, 3);
    handle(SarsaType.expected, 1);
    handle(SarsaType.qlearning, 3);
  }
}
