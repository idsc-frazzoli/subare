// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.ValueFunctions;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** produces results on p.64-65:
 * 
 * {0, 0} 3.3
 * {0, 1} 8.8
 * {0, 2} 4.4
 * {0, 3} 5.3
 * {0, 4} 1.5
 * {1, 0} 1.5
 * {1, 1} 3.0
 * {1, 2} 2.3
 * {1, 3} 1.9
 * {1, 4} 0.5
 * {2, 0} 0.1
 * {2, 1} 0.7
 * {2, 2} 0.7
 * {2, 3} 0.4
 * {2, 4} -0.4
 * {3, 0} -1.0
 * {3, 1} -0.4
 * {3, 2} -0.4
 * {3, 3} -0.6
 * {3, 4} -1.2
 * {4, 0} -1.9
 * {4, 1} -1.3
 * {4, 2} -1.2
 * {4, 3} -1.4
 * {4, 4} -2.0 */
class GridWorld_Ex3_08 {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    Index statesIndex = Index.of(gridWorld.states);
    Tensor result = ValueFunctions.bellmanIteration( //
        gridWorld, //
        new EquiprobablePolicy(gridWorld.actions.length()), //
        statesIndex, DoubleScalar.of(.9), DecimalScalar.of(.0001));
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND));
    }
  }
}
