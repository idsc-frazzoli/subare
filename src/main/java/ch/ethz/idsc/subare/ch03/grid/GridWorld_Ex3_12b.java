// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch03.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EqualPolicy;
import ch.ethz.idsc.subare.core.GreedyPolicy;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.ValueFunctions;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world
 * 
 * produces results on p.71 */
class GridWorld_Ex3_12b {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    Index statesIndex = Index.of(gridWorld.states);
    PolicyInterface policy = new EqualPolicy(gridWorld.actions.length());
    Tensor values = null;
    for (int iters = 0; iters < 5; ++iters) {
      values = ValueFunctions.bellmanIteration( //
          gridWorld, //
          policy, //
          statesIndex, Index.of(gridWorld.actions), DoubleScalar.of(.9), DecimalScalar.of(.0001));
      GreedyPolicy greedyPolicy = GreedyPolicy.build(gridWorld.states, gridWorld.actions, values, gridWorld);
      // greedyPolicy.print(gridWorld.states);
      policy = greedyPolicy;
    }
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + values.get(stateI).map(ROUND));
    }
  }
}
