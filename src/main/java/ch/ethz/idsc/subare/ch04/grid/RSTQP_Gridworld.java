// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** Example 4.1, p.82 */
enum RSTQP_Gridworld {
  ;
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.one());
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("gridworld_qsa_rstqp.gif"), 250)) {
      int batches = 10;
      for (int index = 0; index < batches; ++index) {
        gsw.append(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        Infoline.print(gridworld, index, ref, qsa);
        TabularSteps.batch(gridworld, gridworld, rstqp);
      }
    }
  }
}
