// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.LinearExplorationRate;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;

enum MCES_Wireloop {
  ;
  public static void main(String[] args) throws Exception {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures(name + "L_mces.gif"), 100, TimeUnit.MILLISECONDS)) {
      int batches = 10;
      StateActionCounter sac = new DiscreteStateActionCounter();
      EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, mces.qsa(), sac);
      policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
      for (int index = 0; index < batches; ++index) {
        Infoline infoline = Infoline.print(wireloop, index, ref, mces.qsa());
        for (int count = 0; count < 4; ++count) {
          ExploringStarts.batch(wireloop, policy, mces);
        }
        animationWriter.append(WireloopHelper.render(wireloopRaster, ref, mces.qsa()));
        if (infoline.isLossfree())
          break;
      }
    }
  }
}
