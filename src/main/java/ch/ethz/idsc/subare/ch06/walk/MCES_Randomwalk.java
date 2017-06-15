// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** {0, 0} 0
 * {1, 0} 0.24
 * {2, 0} 0.41
 * {3, 0} 0.54
 * {4, 0} 0.74
 * {5, 0} 0.87
 * {6, 0} 0 */
class MCES_Randomwalk {
  public static void main(String[] args) throws Exception {
    Randomwalk randomwalk = new Randomwalk();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(randomwalk);
    int EPISODES = 1000;
    for (int count = 0; count < EPISODES; ++count) {
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(randomwalk, mces.qsa(), RealScalar.of(.1));
      if (count == 0) {
        boolean equals = Policies.equals(randomwalk, policyInterface, new EquiprobablePolicy(randomwalk));
        if (!equals)
          throw new RuntimeException();
      }
      ExploringStarts.batch(randomwalk, policyInterface, mces);
    }
    DiscreteQsa qsa = mces.qsa();
    qsa.print(Round._2);
  }
}
