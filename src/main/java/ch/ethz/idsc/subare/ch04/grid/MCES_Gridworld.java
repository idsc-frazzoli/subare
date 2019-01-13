// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

/** Example 4.1, p.82 */
enum MCES_Gridworld {
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    try (AnimationWriter animationWriter = AnimationWriter.of(HomeDirectory.Pictures("gridworld_qsa_mces.gif"), 250)) {
      final int batches = 20;
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gridworld, index, ref, mces.qsa());
        for (int count = 0; count < 1; ++count) {
          ExploringStarts.batch(gridworld, policy, mces);
        }
        animationWriter.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), mces.qsa(), ref));
      }
    }
  }
}
