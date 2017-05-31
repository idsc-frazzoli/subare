// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire6";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    PolicyInterface policyInterface = new EquiprobablePolicy(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        wireloop, policyInterface, wireloop, RealScalar.ONE, RealScalar.of(.15));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_qsa_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      mces.simulate(20);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, mces.qsa())));
    }
    gsw.close();
  }
}
