// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

enum IPE_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Policy policy = EquiprobablePolicy.create(wireloop);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        wireloop, policy);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "_ipe_iteration.gif"), 200, TimeUnit.MILLISECONDS)) {
      for (int count = 0; count < 20; ++count) {
        System.out.println(count);
        animationWriter.write(StateRasters.vs_rescale(wireloopRaster, ipe.vs()));
        for (int ep = 0; ep < 5; ++ep)
          ipe.step();
      }
    }
  }
}
