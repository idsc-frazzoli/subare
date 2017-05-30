// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class AVI_Wireloop {
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }

  public static void main(String[] args) throws Exception {
    Wireloop wireloop = WireloopHelper.create("wire7", AVI_Wireloop::id_x);
    System.out.println(Array.zeros(1, 2));
    ActionValueIteration avi = new ActionValueIteration(wireloop, wireloop, RealScalar.ONE);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/wireloop_avi_iteration.gif"), 250);
    for (int count = 0; count < 20; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, avi.qsa())));
      avi.step();
    }
    gsw.close();
    // Tensor image = WireloopHelper.render(wireloop, avi.qsa());
    // BufferedImage bufferedImage = ImageFormat.of(image);
    // Export.of(UserHome.file("Pictures/wireloop_avi.png"), image);
  }
}
