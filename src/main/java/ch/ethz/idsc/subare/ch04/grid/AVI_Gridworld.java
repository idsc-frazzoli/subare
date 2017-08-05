// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 * 
 * Example 4.1, p.82
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
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    ActionValueIteration avi = new ActionValueIteration(gridworld);
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_qsa_avi.gif"), 250);
    for (int count = 0; count < 7; ++count) {
      gsw.append(StateActionRasters.qsa(gridworldRaster, DiscreteValueFunctions.rescaled(avi.qsa())));
      avi.step();
    }
    gsw.append(StateActionRasters.qsa(gridworldRaster, DiscreteValueFunctions.rescaled(avi.qsa())));
    gsw.close();
    // ---
    avi.qsa().print();
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    dvs.print();
  }
}
