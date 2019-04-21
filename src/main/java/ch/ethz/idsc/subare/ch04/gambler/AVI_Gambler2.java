// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma
 * 
 * visualizes each pass of the action value iteration */
/* package */ enum AVI_Gambler2 {
  ;
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    ActionValueIteration avi = ActionValueIteration.of(gambler);
    try (AnimationWriter animationWriter = AnimationWriter.of(HomeDirectory.Pictures("gambler_qsa_avi.gif"), 500)) {
      for (int index = 0; index < 13; ++index) {
        DiscreteQsa qsa = avi.qsa();
        Infoline.print(gambler, index, ref, qsa);
        animationWriter.append(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref));
        avi.step();
      }
      animationWriter.append(ImageFormat.of(StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), avi.qsa(), ref)));
    }
  }
}
