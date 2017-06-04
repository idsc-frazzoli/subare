// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.ConstantAlphaMonteCarloVs;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/**  */
class CAMC_Gridworld {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    ConstantAlphaMonteCarloVs camc = new ConstantAlphaMonteCarloVs(gridworld);
    camc.setAlpha(RealScalar.of(.3));
    // MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_camc.gif"), 100);
    final int EPISODES = 50;
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      // Scalar error = DiscreteQsas.distance(camc.vs(), ref);
      // + " " + error.map(ROUND)
      System.out.println(index);
      for (int count = 0; count < 20; ++count) {
        PolicyInterface policyInterface = //
            new EquiprobablePolicy(gridworld);
        // EGreedyPolicy.bestEquiprobable(gridworld, camc.vs(), epsilon.Get(index));
        ExploringStartsBatch.apply(gridworld, camc, policyInterface);
      }
      gsw.append(ImageFormat.of(GridworldHelper.render(gridworld, camc.vs())));
    }
    gsw.close();
  }
}
