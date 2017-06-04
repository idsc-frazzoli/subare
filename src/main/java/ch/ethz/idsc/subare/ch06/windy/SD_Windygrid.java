// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines q(s,a) function for equiprobable "random" policy */
class SD_Windygrid {
  static void handle(StepDigestType type) {
    System.out.println(type);
    Windygrid windygrid = Windygrid.createFour();
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    System.out.println(qsa.size());
    for (int c = 0; c < 10; ++c) {
      System.out.println(c);
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(windygrid, qsa, RealScalar.of(.1));
      StepDigest stepDigest = type.supply(windygrid, qsa, RealScalar.of(.1), policyInterface);
      for (int count = 0; count < 10; ++count)
        ExploringStartsBatch.apply(windygrid, stepDigest, policyInterface);
    }
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }

  public static void main(String[] args) {
    handle(StepDigestType.qlearning);
  }
}
