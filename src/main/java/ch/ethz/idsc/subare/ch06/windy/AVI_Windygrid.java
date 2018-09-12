// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.Export;

/** action value iteration for cliff walk */
enum AVI_Windygrid {
  ;
  public static void main(String[] args) throws Exception {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    Export.of(UserHome.Pictures("windygrid_qsa_avi.png"), //
        StateActionRasters.qsa_rescaled(windygridRaster, ref));
    ActionValueIteration avi = ActionValueIteration.of(windygrid);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("windygrid_qsa_avi.gif"), 250)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(windygrid, index, ref, avi.qsa());
        gsw.append(StateActionRasters.qsaLossRef(windygridRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
    }
    // TODO extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    DiscreteUtils.print(vs);
    Policy policy = GreedyPolicy.of(windygrid, ref);
    Policies.print(policy, windygrid.states());
  }
}
