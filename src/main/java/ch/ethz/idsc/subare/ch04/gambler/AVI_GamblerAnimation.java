// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.concurrent.TimeUnit;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.core.util.gfx.StateActionRasters;
import ch.ethz.idsc.tensor.io.AnimationWriter;
import ch.ethz.idsc.tensor.io.GifAnimationWriter;
import ch.ethz.idsc.tensor.io.HomeDirectory;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma
 * 
 * visualizes each pass of the action value iteration */
/* package */ enum AVI_GamblerAnimation {
  ;
  public static void main(String[] args) throws Exception {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    ActionValueIteration avi = ActionValueIteration.of(gamblerModel);
    try (AnimationWriter animationWriter = //
        new GifAnimationWriter(HomeDirectory.Pictures("gambler_qsa_avi.gif"), 500, TimeUnit.MILLISECONDS)) {
      for (int index = 0; index < 13; ++index) {
        DiscreteQsa qsa = avi.qsa();
        Infoline.print(gamblerModel, index, ref, qsa);
        animationWriter.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), qsa, ref));
        avi.step();
      }
      animationWriter.write(ImageFormat.of(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), avi.qsa(), ref)));
    }
  }
}
