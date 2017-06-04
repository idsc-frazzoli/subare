// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Cliffwalk {
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/cliffwalk_qsa_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(cliffwalk, mces.qsa(), RealScalar.of(.1));
      ExploringStartsBatch.apply(cliffwalk, mces, policyInterface);
      gsw.append(ImageFormat.of(CliffwalkHelper.render(cliffwalk, mces.qsa())));
    }
    gsw.close();
  }
}
