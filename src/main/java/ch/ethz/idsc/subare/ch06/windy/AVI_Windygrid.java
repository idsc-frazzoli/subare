// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
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
enum AVI_Windygrid {
  ;
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    Export.of(HomeDirectory.Pictures("windygrid_qsa_avi.png"), //
        StateActionRasters.qsa_rescaled(windygridRaster, ref));
    ActionValueIteration avi = ActionValueIteration.of(windygrid);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("windygrid_qsa_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(windygrid, index, ref, avi.qsa());
        animationWriter.append(StateActionRasters.qsaLossRef(windygridRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
    }
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, ref, null);
    Policies.print(policy, windygrid.states());
  }
}
