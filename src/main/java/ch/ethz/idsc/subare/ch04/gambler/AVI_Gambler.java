// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Put;

/** action value iteration for gambler's dilemma */
class AVI_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    Export.of(UserHome.file("Pictures/gambler_qsa_avi.png"), GamblerHelper.qsaPolicyRef(gambler, ref, ref));
    Export.of(UserHome.file("Pictures/gambler_qsa_avi2.png"), GamblerHelper.qsaPolicy(gambler, ref));
    DiscreteVs dvs = DiscreteUtils.createVs(gambler, ref);
    Put.of(UserHome.file("ex403_qsa_values"), dvs.values());
    System.out.println("done.");
  }
}
