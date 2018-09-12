// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** Example 4.1, p.82 */
enum MCES_Gridworld {
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_qsa_mces.gif"), 250)) {
      final int batches = 20;
      Tensor epsilon = Subdivide.of(.2, .05, batches);
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gridworld, index, ref, mces.qsa());
        for (int count = 0; count < 1; ++count) {
          Policy policy = new EGreedyPolicy(gridworld, mces.qsa(), epsilon.Get(index));
          ExploringStarts.batch(gridworld, policy, mces);
        }
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), mces.qsa(), ref));
      }
    }
  }
}
