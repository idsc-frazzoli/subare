// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class AVI_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire6";
    Scalar stepCost = RealScalar.of(-.5);
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x, stepCost);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    ActionValueIteration avi = new ActionValueIteration(wireloop);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L" + stepCost + "_avi.gif"), 250);
    for (int count = 0; count < 44; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, ref, avi.qsa())));
      avi.step();
    }
    gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, ref, avi.qsa())));
    gsw.close();
  }
}
