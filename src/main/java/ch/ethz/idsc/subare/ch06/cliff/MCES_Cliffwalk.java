// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;

/** monte carlo is bad in this example, since the steep negative reward biases most episodes */
// TODO this does not really converge at all
enum MCES_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("cliffwalk_qsa_mces.gif"), 100, TimeUnit.MILLISECONDS)) {
      int batches = 100;
      for (int index = 0; index < batches; ++index) {
        Infoline.print(cliffwalk, index, ref, mces.qsa());
        for (int count = 0; count < 10; ++count) {
          StateActionCounter sac = new DiscreteStateActionCounter();
          PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(cliffwalk, mces.qsa(), sac);
          ExploringStarts.batch(cliffwalk, policy, mces);
        }
        animationWriter.write(StateActionRasters.qsaLossRef(cliffwalkRaster, mces.qsa(), ref));
      }
    }
  }
}
