// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 * 
 * produces results on p.71 */
class GridWorld_Ex4_01a {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    Index statesIndex = Index.build(gridWorld.states);
    Tensor values = new ValueIteration(gridWorld, RealScalar.ONE).untilBelow( //
        DecimalScalar.of(.0001));
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobable(gridWorld, values);
    greedyPolicy.print(gridWorld.states());
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + values.get(stateI).map(ROUND));
    }
  }
}
