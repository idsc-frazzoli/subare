// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

/** action value iteration for cliff walk */
enum AVI_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    Export.of(HomeDirectory.Pictures("cliffwalk_qsa_avi.png"), //
        StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(cliffwalk);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("cliffwalk_qsa_avi.gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(cliffwalk, index, ref, avi.qsa());
        animationWriter.append(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
      animationWriter.append(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
    }
    DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(cliffwalk, ref, null);
    Policies.print(policy, cliffwalk.states());
  }
}
