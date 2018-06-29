// code by jph
// inspired by Shangtong Zhang
package ch.ethz.idsc.subare.ch00.fish;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** action value iteration for cliff walk */
enum AVI_Fishfarm {
  ;
  public static void main(String[] args) throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster fishfarmRaster = new FishfarmRaster(fishfarm);
    DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    // Export.of(UserHome.Pictures("cliffwalk_qsa_avi.png"), //
    // StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(fishfarm);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("fishfarm_qsa_avi.gif"), 200)) {
      for (int index = 0; index < 20; ++index) {
        Infoline infoline = Infoline.print(fishfarm, index, ref, avi.qsa());
        gsw.append(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
        avi.step();
        if (infoline.isErrorFree())
          break;
      }
      gsw.append(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
      gsw.append(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
      gsw.append(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
    }
    // DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    // vs.print();
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, ref);
    // Policies.print(policy, cliffwalk.states());
  }
}
