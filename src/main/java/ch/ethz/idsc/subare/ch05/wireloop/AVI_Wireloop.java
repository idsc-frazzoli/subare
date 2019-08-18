// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

enum AVI_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    ActionValueIteration avi = ActionValueIteration.of(wireloop);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "L_avi.gif"), 250, TimeUnit.MILLISECONDS)) {
      int batches = 50;
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, avi.qsa());
        animationWriter.append(WireloopHelper.render(wireloopRaster, ref, avi.qsa()));
        avi.step();
        if (infoline.isLossfree())
          break;
      }
      animationWriter.append(WireloopHelper.render(wireloopRaster, ref, avi.qsa()));
    }
  }
}
