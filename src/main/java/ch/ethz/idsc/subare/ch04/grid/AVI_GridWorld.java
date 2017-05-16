// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 *
 * produces results on p.71
 * chapter 4, example 1
 * 
 * {0, 0} 0
 * {0, 1} -1.0
 * {0, 2} -2.0
 * {0, 3} -3.0
 * {1, 0} -1.0
 * {1, 1} -2.0
 * {1, 2} -3.0
 * {1, 3} -2.0
 * {2, 0} -2.0
 * {2, 1} -3.0
 * {2, 2} -2.0
 * {2, 3} -1.0
 * {3, 0} -3.0
 * {3, 1} -2.0
 * {3, 2} -1.0
 * {3, 3} 0 */
class AVI_GridWorld {
  public static void main(String[] args) {
    GridWorld gridWorld = new GridWorld();
    ActionValueIteration avi = new ActionValueIteration(gridWorld, gridWorld, RealScalar.ONE);
    avi.untilBelow(DecimalScalar.of(.0001));
    avi.qsa().print(Round.toMultipleOf(DecimalScalar.of(.1)));
    DiscreteVs dvs = DiscreteUtils.createVs(gridWorld, avi.qsa());
    dvs.print();
  }
}
