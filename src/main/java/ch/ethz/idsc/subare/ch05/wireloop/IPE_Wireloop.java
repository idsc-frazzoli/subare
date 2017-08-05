// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

class IPE_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Policy policy = new EquiprobablePolicy(wireloop);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        wireloop, policy);
    AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures(name + "_ipe_iteration.gif"), 200);
    for (int count = 0; count < 20; ++count) {
      System.out.println(count);
      gsw.append(StateRasters.vs_rescale(wireloopRaster, ipe.vs()));
      for (int ep = 0; ep < 5; ++ep)
        ipe.step();
    }
    gsw.close();
  }
}
