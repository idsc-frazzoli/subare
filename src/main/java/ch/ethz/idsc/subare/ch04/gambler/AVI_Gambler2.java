// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** action value iteration for gambler's dilemma */
class AVI_Gambler2 {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    ActionValueIteration avi = new ActionValueIteration(gambler, gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_avi_iteration.gif"), 500);
    GifSequenceWriter gsp = GifSequenceWriter.of(UserHome.file("Pictures/gambler_avi_policy.gif"), 500);
    for (int count = 0; count < 13; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(GamblerHelper.render(gambler, avi.qsa())));
      {
        GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(gambler, avi.qsa());
        gsp.append(ImageFormat.of(GamblerHelper.render(gambler, greedyPolicy)));
      }
      avi.step();
    }
    gsw.append(ImageFormat.of(GamblerHelper.render(gambler, avi.qsa())));
    {
      GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(gambler, avi.qsa());
      gsp.append(ImageFormat.of(GamblerHelper.render(gambler, greedyPolicy)));
    }
    gsw.close();
    gsp.close();
  }
}
