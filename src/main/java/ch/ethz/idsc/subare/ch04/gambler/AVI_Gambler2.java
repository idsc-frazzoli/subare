// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.Infoline;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma */
class AVI_Gambler2 {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    ActionValueIteration avi = new ActionValueIteration(gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("gambler_qsa_avi.gif"), 500);
    for (int index = 0; index < 13; ++index) {
      DiscreteQsa qsa = avi.qsa();
      Infoline.print(gambler, index, ref, qsa);
      gsw.append(ImageFormat.of(GamblerHelper.qsaPolicyRef(gambler, qsa, ref)));
      avi.step();
    }
    gsw.append(ImageFormat.of(GamblerHelper.qsaPolicyRef(gambler, avi.qsa(), ref)));
    gsw.close();
  }
}
