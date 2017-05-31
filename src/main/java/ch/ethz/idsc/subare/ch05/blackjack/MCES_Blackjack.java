// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class MCES_Blackjack {
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    PolicyInterface policyInterface = new EquiprobablePolicy(blackjack);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        blackjack, policyInterface, blackjack, RealScalar.of(.1));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/blackjack_qsa_mces.gif"), 200);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      mces.simulate(2000);
      gsw.append(ImageFormat.of(BlackjackHelper.render(blackjack, mces.qsa())));
    }
    gsw.close();
  }
}
