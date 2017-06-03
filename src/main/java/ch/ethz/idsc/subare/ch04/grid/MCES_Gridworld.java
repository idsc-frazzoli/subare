// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Gridworld {
  public static void main(String[] args) throws Exception {
    Gridworld gridworld = new Gridworld();
    PolicyInterface policyInterface = new EquiprobablePolicy(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld, policyInterface);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gridworld_qsa_mces.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      mces.setExplorationProbability(RealScalar.of(.1));
      // mces.simulate(1);
      ExploringStartBatch.apply(gridworld, mces, policyInterface);
      gsw.append(ImageFormat.of(GridworldHelper.render(gridworld, mces.qsa())));
    }
    gsw.close();
  }
}
