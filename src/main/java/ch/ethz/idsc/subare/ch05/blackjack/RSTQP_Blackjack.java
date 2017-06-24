// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.TabularSteps;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;

/** finding optimal policy to stay or hit
 * 
 * Random1StepTabularQPlanning does not seem to work on blackjack */
class RSTQP_Blackjack {
  public static void main(String[] args) throws Exception {
    Blackjack blackjack = new Blackjack();
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    Random1StepTabularQPlanning rstqp = new Random1StepTabularQPlanning( //
        blackjack, qsa, DefaultLearningRate.of(5, 0.51));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.Pictures("blackjack_rstqp.gif"), 250);
    int batches = 60;
    for (int index = 0; index < batches; ++index) {
      for (int count = 0; count < 100; ++count)
        TabularSteps.batch(blackjack, blackjack, rstqp);
      gsw.append(BlackjackHelper.joinAll(blackjack, qsa));
    }
    gsw.close();
  }
}
