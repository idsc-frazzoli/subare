// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -9.00
 * {0, 2} -19.90
 * {0, 3} -20.52
 * {1, 0} -13.61
 * {1, 1} -16.52
 * {1, 2} -17.52
 * {1, 3} -19.84
 * {2, 0} -16.20
 * {2, 1} -17.77
 * {2, 2} -19.94
 * {2, 3} -11.45
 * {3, 0} -21.01
 * {3, 1} -19.68
 * {3, 2} -18.52
 * {3, 3} 0 */
public class TTD0_Gridworld {
  public static void main(String[] args) {
    Gridworld gridWorld = new Gridworld();
    DiscreteVs vs = DiscreteVs.build(gridWorld);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        vs, gridWorld.gamma(), RealScalar.of(.5));
    PolicyInterface policyInterface = new EquiprobablePolicy(gridWorld);
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(gridWorld, policyInterface, ttd0);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
