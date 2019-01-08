// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.Put;

/** action value iteration for gambler's dilemma
 * 
 * visualizes the exact optimal policy */
enum AVI_Gambler {
  ;
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    Export.of(HomeDirectory.Pictures("gambler_qsa_avi.png"), //
        StateActionRasters.qsaPolicy(gamblerRaster, ref));
    DiscreteVs vs = DiscreteUtils.createVs(gambler, ref);
    Put.of(HomeDirectory.file("ex403_vs_values"), vs.values());
    System.out.println("done.");
  }
}
