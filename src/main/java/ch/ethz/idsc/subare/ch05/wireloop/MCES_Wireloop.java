// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Wireloop {
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }

  public static void main(String[] args) throws Exception {
    Wireloop wireloop = WireloopHelper.create("wire6", MCES_Wireloop::id_x);
    System.out.println(Array.zeros(1, 2));
    PolicyInterface policyInterface = new EquiprobablePolicy(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        wireloop, policyInterface, wireloop, RealScalar.ONE, RealScalar.of(.15));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/wireloop_qsa_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      mces.simulate(20);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, mces.qsa())));
    }
    gsw.close();
    // ActionValueIteration avi = new ActionValueIteration(wireloop, wireloop, RealScalar.ONE);
    // GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/wireloop_avi_iteration.gif"), 250);
    // for (int count = 0; count < 20; ++count) {
    // System.out.println(count);
    // gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, avi.qsa())));
    // avi.step();
    // }
    // gsw.close();
    // Tensor image = WireloopHelper.render(wireloop, avi.qsa());
    // BufferedImage bufferedImage = ImageFormat.of(image);
    // Export.of(UserHome.file("Pictures/wireloop_avi.png"), image);
  }
}
