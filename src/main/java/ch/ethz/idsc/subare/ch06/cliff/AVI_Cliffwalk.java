// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;

/** action value iteration for cliff walk */
class AVI_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    Export.of(UserHome.Pictures("cliffwalk_qsa_avi.png"), //
        StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = new ActionValueIteration(cliffwalk);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("cliffwalk_qsa_avi.gif"), 200);
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.print(cliffwalk, index, ref, avi.qsa());
      gsw.append(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    gsw.append(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
    gsw.close();
    DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    vs.print();
    Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, ref);
    Policies.print(policy, cliffwalk.states());
  }
}
