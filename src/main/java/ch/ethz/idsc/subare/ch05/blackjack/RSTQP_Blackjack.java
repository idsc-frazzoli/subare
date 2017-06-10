// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
class RSTQP_Blackjack {
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        blackjack, blackjack, qsa);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/blackjack_rstqp.gif"), 250);
    int EPISODES = 60;
    Tensor epsilon = Subdivide.of(.9, .05, EPISODES);
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index + " " + epsilon.Get(index));
      rstqp.setLearningRate(epsilon.Get(index));
      for (int count = 0; count < 100; ++count)
        rstqp.batch();
      gsw.append(ImageFormat.of(BlackjackHelper.joinAll(blackjack, qsa)));
    }
    gsw.close();
  }
}
