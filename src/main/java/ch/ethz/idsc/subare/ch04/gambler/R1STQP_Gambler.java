// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.alg.Random1StepTabularQPlanning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

// R1STQP algorithm is not suited for gambler's dilemma
class R1STQP_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    Random1StepTabularQPlanning qLearning = new Random1StepTabularQPlanning( //
        gambler, gambler, qsa, RealScalar.ONE, RealScalar.of(.1));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_r1stqp.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      for (int c1 = 0; c1 < 100; ++c1)
        qLearning.step();
      gsw.append(ImageFormat.of(GamblerHelper.render(gambler, qsa)));
    }
    gsw.close();
  }
}
