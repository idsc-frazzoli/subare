// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.Digits;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** Example 4.1, p.82 */
class RSTQP_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        gridworld, qsa, DefaultLearningRate.of(5, 1.0)); // TODO try learning rate
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gridworld_qsa_rstqp.gif"), 150);
    int EPISODES = 30;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = TensorValuesUtils.distance(qsa, ref);
      System.out.println(index + " " + error.map(Digits._1));
      TabularSteps.batch(gridworld, gridworld, rstqp);
      gsw.append(ImageFormat.of(GridworldHelper.joinAll(gridworld, qsa, ref)));
    }
    gsw.close();
  }
}
