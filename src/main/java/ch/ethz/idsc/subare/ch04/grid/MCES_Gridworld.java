// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** Example 4.1, p.82 */
class MCES_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gridworld_qsa_mces.gif"), 100);
    final int EPISODES = 50;
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteValueFunctions.distance(mces.qsa(), ref);
      System.out.println(index + " " + error.map(Round._1));
      for (int count = 0; count < 20; ++count) {
        Policy policy = EGreedyPolicy.bestEquiprobable(gridworld, mces.qsa(), epsilon.Get(index));
        ExploringStarts.batch(gridworld, policy, mces);
      }
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, mces.qsa(), ref)));
    }
    gsw.close();
  }
}
