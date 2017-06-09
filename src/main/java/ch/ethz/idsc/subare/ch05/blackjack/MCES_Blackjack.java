// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.ActionValueStatistics;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
class MCES_Blackjack {
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/blackjack_mces.gif"), 250);
    int EPISODES = 10; // 40
    Tensor epsilon = Subdivide.of(.2, .05, EPISODES);
    int episodes = 0;
    ActionValueStatistics avs = new ActionValueStatistics(blackjack);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index + " " + epsilon.Get(index));
      for (int count = 0; count < EPISODES; ++count) {
        PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(blackjack, mces.qsa(), epsilon.Get(index));
        episodes += //
            ExploringStarts.batchWithReplay(blackjack, policyInterface, mces, avs);
      }
      gsw.append(ImageFormat.of(BlackjackHelper.joinAll(blackjack, mces.qsa())));
      System.out.println(episodes + " " + avs.coverage());
    }
    gsw.close();
    boolean isComplete = avs.isComplete();
    if (isComplete) {
      ActionValueIteration avi = new ActionValueIteration(blackjack, avs);
      avi.untilBelow(RealScalar.of(.01), 4);
      Export.of(UserHome.file("Pictures/blackjack_avi.png"), BlackjackHelper.joinAll(blackjack, avi.qsa()));
    } else {
      System.out.println("statistics are incomplete!");
    }
  }
}
