// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch08.maze;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** action value iteration for cliff walk */
enum AVI_Dynamaze {
  ;
  public static void main(String[] args) throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    // Export.of(UserHome.Pictures("dynamaze_qsa_avi.png"), //
    // DynamazeHelper.render(windygrid, DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(dynamaze);
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures(name + "_qsa_avi.gif"), 250);
    for (int index = 0; index < 50; ++index) {
      Infoline infoline = Infoline.print(dynamaze, index, ref, avi.qsa());
      gsw.append(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    // gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, avi.qsa(), ref)));
    gsw.close();
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(dynamaze, ref);
    DiscreteUtils.print(vs);
    Policy policy = GreedyPolicy.bestEquiprobable(dynamaze, ref);
    Policies.print(policy, dynamaze.states());
  }
}
