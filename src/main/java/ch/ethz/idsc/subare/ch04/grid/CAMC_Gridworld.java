// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.ConstantAlphaMonteCarloVs;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/**  */
class CAMC_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    ConstantAlphaMonteCarloVs camc = new ConstantAlphaMonteCarloVs( //
        gridworld, DefaultLearningRate.of(3, .51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gridworld_qsa_camc.gif"), 100);
    final int EPISODES = 50;
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      for (int count = 0; count < 20; ++count) {
        Policy policy = //
            new EquiprobablePolicy(gridworld);
        // EGreedyPolicy.bestEquiprobable(gridworld, camc.vs(), epsilon.Get(index));
        ExploringStarts.batch(gridworld, policy, camc);
      }
      gsw.append(ImageFormat.of(GridworldHelper.render(gridworld, camc.vs())));
    }
    gsw.close();
  }
}
