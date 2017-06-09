// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** tabular temporal difference (0) to learn value of states
 * 
 * 0 0
 * 1 0.10
 * 2 0.27
 * 3 0.47
 * 4 0.67
 * 5 0.90
 * 6 0 */
class TTD0_Randomwalk {
  public static void main(String[] args) {
    Randomwalk randomwalk = new Randomwalk();
    DiscreteVs vs = DiscreteVs.build(randomwalk);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        vs, randomwalk.gamma(), RealScalar.of(.1));
    PolicyInterface policyInterface = new EquiprobablePolicy(randomwalk);
    for (int count = 0; count < 1000; ++count)
      ExploringStarts.batch(randomwalk, policyInterface, ttd0);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
