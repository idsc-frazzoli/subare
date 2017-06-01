// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma */
class AVI_Gambler2 {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_avi.gif"), 500);
    for (int count = 0; count < 13; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, avi.qsa())));
      avi.step();
    }
    gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, avi.qsa())));
    gsw.close();
  }
}
