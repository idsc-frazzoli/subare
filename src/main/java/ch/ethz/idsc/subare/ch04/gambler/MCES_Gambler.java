// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.sca.Round;

enum MCES_Gambler {
  ;
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, mces.qsa(), sac);
    try (AnimationWriter gsw = AnimationWriter.of(HomeDirectory.Pictures("gambler_qsa_mces.gif"), 200)) {
      int batches = 20;
      for (int index = 0; index < batches; ++index) {
        Infoline.print(gambler, index, ref, mces.qsa());
        ExploringStarts.batch(gambler, policy, mces);
        gsw.append(StateActionRasters.qsaPolicyRef(gamblerRaster, mces.qsa(), ref));
      }
    }
    System.out.println("done");
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    DiscreteUtils.print(discreteVs, Round._2);
  }
}
