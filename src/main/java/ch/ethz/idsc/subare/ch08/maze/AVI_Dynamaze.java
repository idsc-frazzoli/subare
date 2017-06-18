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
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for cliff walk */
class AVI_Dynamaze {
  public static void main(String[] args) throws Exception {
    Dynamaze dynamaze = DynamazeHelper.create("maze2");
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    // Export.of(UserHome.Pictures("dynamaze_qsa_avi.png"), //
    // DynamazeHelper.render(windygrid, DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = new ActionValueIteration(dynamaze);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("dynamaze_qsa_avi.gif"), 250);
    for (int index = 0; index < 20; ++index) {
      Infoline.print(dynamaze, index, ref, avi.qsa());
      DiscreteVs vs = DiscreteUtils.createVs(dynamaze, avi.qsa());
      gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, vs)));
      avi.step();
    }
    // gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, avi.qsa(), ref)));
    gsw.close();
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(dynamaze, ref);
    vs.print();
    Policy policy = GreedyPolicy.bestEquiprobable(dynamaze, ref);
    Policies.print(policy, dynamaze.states());
  }
}
