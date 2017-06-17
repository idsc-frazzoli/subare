// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures(name + "L_mces.gif"), 100);
    int EPISODES = 10;
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      Infoline.print(wireloop, index, ref, mces.qsa());
      for (int count = 0; count < 4; ++count) {
        Policy policy = EGreedyPolicy.bestEquiprobable(wireloop, mces.qsa(), epsilon.Get(index));
        ExploringStarts.batch(wireloop, policy, mces);
      }
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, ref, mces.qsa())));
    }
    gsw.close();
  }
}
