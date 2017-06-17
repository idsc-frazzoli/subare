// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** Example 4.1, p.82 */
class RSTQP_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        gridworld, qsa, ConstantLearningRate.one());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gridworld_qsa_rstqp.gif"), 150);
    int EPISODES = 30;
    for (int index = 0; index < EPISODES; ++index) {
      Infoline.print(gridworld, index, ref, qsa);
      TabularSteps.batch(gridworld, gridworld, rstqp);
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa, ref)));
    }
    gsw.close();
  }
}
