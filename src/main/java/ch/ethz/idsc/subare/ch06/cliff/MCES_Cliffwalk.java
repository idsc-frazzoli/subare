// code by jph
package ch.ethz.idsc.subare.ch06.cliff;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** monte carlo is bad in this example, since the steep negative reward biases most episodes */
// TODO this does not really converge at all
enum MCES_Cliffwalk {
  ;
  public static void main(String[] args) throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("cliffwalk_qsa_mces.gif"), 100)) {
      int batches = 100;
      for (int index = 0; index < batches; ++index) {
        Infoline.print(cliffwalk, index, ref, mces.qsa());
        for (int count = 0; count < 10; ++count) {
          Policy policy = //
              EGreedyPolicy.bestEquiprobable(cliffwalk, mces.qsa(), RealScalar.of(.4));
          ExploringStarts.batch(cliffwalk, policy, mces);
        }
        gsw.append(StateActionRasters.qsaLossRef(cliffwalkRaster, mces.qsa(), ref));
      }
    }
  }
}
