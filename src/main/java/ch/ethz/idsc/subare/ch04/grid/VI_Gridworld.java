// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.io.Export;

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
class VI_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldStateRaster = new GridworldRaster(gridworld);
    ValueIteration vi = new ValueIteration(gridworld, gridworld);
    vi.untilBelow(DecimalScalar.of(.0001));
    vi.vs().print();
    Policy policy = GreedyPolicy.bestEquiprobable(gridworld, vi.vs());
    Policies.print(policy, gridworld.states());
    Export.of(UserHome.Pictures("gridworld_vs_vi.png"), //
        StateRasters.vs(gridworldStateRaster, DiscreteValueFunctions.rescaled(vi.vs())));
    // GridworldHelper.render(gridworld, vi.vs())
  }
}
