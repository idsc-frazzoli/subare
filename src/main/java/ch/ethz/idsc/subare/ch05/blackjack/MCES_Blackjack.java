// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
class MCES_Blackjack {
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    PolicyInterface policyInterface = new EquiprobablePolicy(blackjack);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack, policyInterface);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/blackjack_mces.gif"), 250);
    int EPISODES = 40;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar epsilon = RealScalar.of(.3 * (EPISODES - index) / EPISODES);
      System.out.println(index + " " + epsilon);
      mces.setExplorationProbability(epsilon);
      ExploringStartsBatch.apply(blackjack, mces, policyInterface);
      gsw.append(ImageFormat.of(BlackjackHelper.joinAll(blackjack, mces.qsa())));
    }
    gsw.close();
  }
}
