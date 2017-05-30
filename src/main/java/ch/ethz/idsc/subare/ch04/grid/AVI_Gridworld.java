// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.io.Export;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 *
 * produces results on p.71
 * chapter 4, example 1
 * 
 * {0, 0} 0
 * {0, 1} -1
 * {0, 2} -2
 * {0, 3} -3
 * {1, 0} -1
 * {1, 1} -2
 * {1, 2} -3
 * {1, 3} -2
 * {2, 0} -2
 * {2, 1} -3
 * {2, 2} -2
 * {2, 3} -1
 * {3, 0} -3
 * {3, 1} -2
 * {3, 2} -1
 * {3, 3} 0 */
class AVI_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    ActionValueIteration avi = new ActionValueIteration(gridworld, gridworld);
    avi.untilBelow(DecimalScalar.of(.0001));
    avi.qsa().print();
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    dvs.print();
    Export.of(UserHome.file("Pictures/gridworld_qsa_avi.png"), GridworldHelper.render(gridworld, avi.qsa()));
  }
}
