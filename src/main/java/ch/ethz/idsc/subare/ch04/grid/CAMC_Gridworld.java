// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.EpisodeVsEstimator;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.ConstantAlphaMonteCarloVs;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/**  */
enum CAMC_Gridworld { // TODO this looks like WIP
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    // final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    EpisodeVsEstimator camc = ConstantAlphaMonteCarloVs.create( //
        gridworld, DefaultLearningRate.of(3, .51));
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_vs_camc.gif"), 100);
    final int batches = 50;
    // Tensor epsilon = Subdivide.of(.2, .05, batches);
    for (int index = 0; index < batches; ++index) {
      System.out.println(index);
      for (int count = 0; count < 20; ++count) {
        Policy policy = EquiprobablePolicy.create(gridworld);
        // EGreedyPolicy.bestEquiprobable(gridworld, camc.vs(), epsilon.Get(index));
        ExploringStarts.batch(gridworld, policy, camc);
      }
      gsw.append(StateRasters.vs(gridworldRaster, DiscreteValueFunctions.rescaled(camc.vs())));
    }
    gsw.close();
  }
}
