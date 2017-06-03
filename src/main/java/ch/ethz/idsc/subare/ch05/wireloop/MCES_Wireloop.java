// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Wireloop {
  public static void main(String[] args) throws Exception {
    String name = "wire4";
    Wireloop wireloop = WireloopHelper.create(name, WireloopHelper::id_x);
    PolicyInterface policyInterface = new EquiprobablePolicy(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop, policyInterface);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/" + name + "_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar epsilon = RealScalar.of(.25 * (EPISODES - index) / EPISODES);
      System.out.println(index + " " + epsilon);
      mces.setExplorationProbability(epsilon);
      // mces.simulate(100);
      ExploringStartBatch.apply(wireloop, mces, policyInterface);
      gsw.append(ImageFormat.of(WireloopHelper.render(wireloop, mces.qsa())));
    }
    gsw.close();
  }
}
