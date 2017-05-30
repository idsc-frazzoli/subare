// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class AVI_Wireloop {
  public static void main(String[] args) throws Exception {
    Wireloop wireloop = WireloopHelper.create("wire7", WireloopHelper::id_x);
    ActionValueIteration avi = new ActionValueIteration(wireloop, wireloop);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/wireloop_avi_iteration.gif"), 250);
    for (int count = 0; count < 20; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, avi.qsa())));
      avi.step();
    }
    gsw.close();
  }
}
