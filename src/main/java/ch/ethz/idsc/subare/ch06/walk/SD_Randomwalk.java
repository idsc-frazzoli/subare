// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** determines state action value function q(s,a).
 * initial policy is irrelevant because each state allows only one action.
 * 
 * {0, 0} 0
 * {1, 0} 0.16
 * {2, 0} 0.35
 * {3, 0} 0.47
 * {4, 0} 0.59
 * {5, 0} 0.79
 * {6, 0} 0 */
class SD_Randomwalk {
  static void handle(StepDigestType type) {
    System.out.println(type);
    Randomwalk randomwalk = new Randomwalk();
    DiscreteQsa qsa = DiscreteQsa.build(randomwalk);
    PolicyInterface policyInterface = new EquiprobablePolicy(randomwalk);
    StepDigest stepDigest = type.supply(randomwalk, qsa, RealScalar.of(.1), policyInterface);
    for (int count = 0; count < 1000; ++count)
      ExploringStartsBatch.apply(randomwalk, stepDigest, policyInterface);
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }

  public static void main(String[] args) {
    for (StepDigestType type : StepDigestType.values())
      handle(type);
  }
}
