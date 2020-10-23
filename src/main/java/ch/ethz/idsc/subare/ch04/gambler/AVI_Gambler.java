// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.Put;

/** action value iteration for gambler's dilemma
 * 
 * visualizes the exact optimal policy */
/* package */ enum AVI_Gambler {
  ;
  public static void main(String[] args) throws Exception {
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.35));
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel) {
      @Override
      public int magnify() {
        return 1;
      }
    };
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    Export.of(HomeDirectory.Pictures("gambler_qsa.png"), //
        StateActionRasters.qsa(gamblerRaster, ref));
    Export.of(HomeDirectory.Pictures("gambler_qsa_avi.png"), //
        StateActionRasters.qsaPolicy(gamblerRaster, ref));
    DiscreteVs vs = DiscreteUtils.createVs(gamblerModel, ref);
    Put.of(HomeDirectory.file("ex403_vs_values"), vs.values());
    System.out.println("done.");
  }
}
