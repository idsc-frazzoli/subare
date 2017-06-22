// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class AVI_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    ActionValueIteration avi = new ActionValueIteration(wireloop);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L_avi.gif"), 250);
    int batches = 50;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(wireloop, index, ref, avi.qsa());
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloopRaster, ref, avi.qsa())));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    gsw.append(ImageFormat.of(WireloopHelper.render(wireloopRaster, ref, avi.qsa())));
    gsw.close();
  }
}
