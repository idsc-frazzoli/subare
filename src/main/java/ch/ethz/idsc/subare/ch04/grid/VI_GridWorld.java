// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 *
 * produces results on p.71
 * chapter 4, example 1 */
class VI_GridWorld {
  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    ValueIteration vi = new ValueIteration(gridWorld, RealScalar.ONE);
    vi.untilBelow(DecimalScalar.of(.0001));
    vi.vs().print(Round.toMultipleOf(DecimalScalar.of(.1)));
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(gridWorld, vi.vs().values());
    greedyPolicy.print(gridWorld.states());
  }
}
