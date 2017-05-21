// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;

class QL_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    QLearning qLearning = new QLearning( //
        gambler, new EquiprobablePolicy(gambler), //
        gambler, //
        qsa, RealScalar.ONE, RealScalar.of(.1)); // TODO ask jz
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_qlearn.gif"), 100);
    int EPISODES = 100;
    for (int index = 0; index < EPISODES; ++index) {
      System.out.println(index);
      qLearning.simulate(300);
      gsw.append(ImageFormat.of(GamblerHelper.render(gambler, qsa)));
    }
    gsw.close();
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, qsa);
    Put.of(UserHome.file("ql_gambler"), discreteVs.values());
  }
}
