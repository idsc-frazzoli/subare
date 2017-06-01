// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** tabular temporal difference (0) to learn value of states
 * 
 * 0 0
 * 1 0.11
 * 2 0.36
 * 3 0.64
 * 4 0.79
 * 5 0.90
 * 6 0 */
class TTD0_RandomWalk {
  public static void main(String[] args) {
    RandomWalk randomWalk = new RandomWalk();
    DiscreteVs vs = DiscreteVs.build(randomWalk);
    PolicyInterface policyInterface = new EquiprobablePolicy(randomWalk);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        randomWalk, policyInterface, //
        vs, randomWalk.gamma(), RealScalar.of(.1));
    ttd0.simulate(123);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
