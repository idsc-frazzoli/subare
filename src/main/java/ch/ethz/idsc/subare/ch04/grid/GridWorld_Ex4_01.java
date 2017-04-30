// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.ValueFunctions;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** determines value function for equiprobable "random" policy
 * 
 * produces results on p.83:
 * 
 * {0, 0} 0
 * {0, 1} -14.0
 * {0, 2} -20.0
 * {0, 3} -22.0
 * {1, 0} -14.0
 * {1, 1} -18.0
 * {1, 2} -20.0
 * {1, 3} -20.0
 * {2, 0} -20.0
 * {2, 1} -20.0
 * {2, 2} -18.0
 * {2, 3} -14.0
 * {3, 0} -22.0
 * {3, 1} -20.0
 * {3, 2} -14.0
 * {3, 3} 0 */
class GridWorld_Ex4_01 {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    Index statesIndex = Index.build(gridWorld.states);
    Tensor result = ValueFunctions.bellmanIteration( //
        gridWorld, //
        new EquiprobablePolicy(gridWorld.actions.length()), //
        RealScalar.ONE, DecimalScalar.of(.0001));
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND));
    }
  }
}
