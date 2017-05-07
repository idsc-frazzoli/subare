// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world
 * 
 * produces results on p.71:
 * 
 * {0, 0} 22.0
 * {0, 1} 24.4
 * {0, 2} 22.0
 * {0, 3} 19.4
 * {0, 4} 17.5
 * {1, 0} 19.8
 * {1, 1} 22.0
 * {1, 2} 19.8
 * {1, 3} 17.8
 * {1, 4} 16.0
 * {2, 0} 17.8
 * {2, 1} 19.8
 * {2, 2} 17.8
 * {2, 3} 16.0
 * {2, 4} 14.4
 * {3, 0} 16.0
 * {3, 1} 17.8
 * {3, 2} 16.0
 * {3, 3} 14.4
 * {3, 4} 13.0
 * {4, 0} 14.4
 * {4, 1} 16.0
 * {4, 2} 14.4
 * {4, 3} 13.0
 * {4, 4} 11.7 */
class GridWorld_Ex3_12a {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    Index statesIndex = Index.build(gridWorld.states);
    ValueIteration vi = new ValueIteration(gridWorld, DoubleScalar.of(.9));
    Tensor values = vi.untilBelow( //
        DecimalScalar.of(.0001));
    System.out.println("iterations=" + vi.iterations());
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(gridWorld, values);
    greedyPolicy.print(gridWorld.states());
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + values.get(stateI).map(ROUND));
    }
  }
}
