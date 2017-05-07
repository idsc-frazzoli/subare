// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** {0, 0} 0
 * {0, 1} -14.01
 * {0, 2} -18.77
 * {0, 3} -22.19
 * {1, 0} -12.14
 * {1, 1} -15.75
 * {1, 2} -19.44
 * {1, 3} -21.50
 * {2, 0} -18.98
 * {2, 1} -19.12
 * {2, 2} -16.80
 * {2, 3} -16.13
 * {3, 0} -21.88
 * {3, 1} -20.04
 * {3, 2} -12.08
 * {3, 3} 0 */
public class TTD0_GridWorld {
  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    PolicyInterface policyInterface = new EquiprobablePolicy(gridWorld);
    DiscreteVs vs = DiscreteVs.build(gridWorld);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        gridWorld, policyInterface, //
        vs, RealScalar.ONE, RealScalar.of(.1));
    ttd0.simulate(10230);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
