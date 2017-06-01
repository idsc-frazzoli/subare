// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;

class QL_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qlearn.gif"), 500);
    int EPISODES = 10;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar scalar = RealScalar.of(.01 + .3 * (EPISODES - index) / EPISODES);
      System.out.println(index + " " + scalar);
      PolicyInterface policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, scalar);
      QLearning qLearning = new QLearning( //
          gambler, policyInterface, gambler, qsa, scalar);
      qLearning.simulate(5000);
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, qsa)));
    }
    gsw.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, qsa);
    Put.of(UserHome.file("ql_gambler"), discreteVs.values());
  }
}
