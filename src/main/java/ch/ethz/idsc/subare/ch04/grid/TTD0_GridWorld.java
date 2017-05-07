// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.TabularTemporalDifference0;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
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
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) {
    GridWorld randomWalk = new GridWorld();
    PolicyInterface policyInterface = new EquiprobablePolicy(randomWalk);
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        randomWalk, policyInterface, RealScalar.of(.1), RealScalar.ONE, randomWalk);
    Tensor result = ttd0.simulate(10230);
    Index statesIndex = Index.build(randomWalk.states);
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND));
    }
  }
}
