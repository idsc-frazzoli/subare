// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

class MCES_Gambler {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.01));

  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gambler);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_mces.gif"), 200);
    int EPISODES = 20;
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(mces.qsa(), ref);
      System.out.println(index + " " + error.map(ROUND));
      for (int count = 0; count < 2; ++count) {
        PolicyInterface policyInterface = //
            EGreedyPolicy.bestEquiprobable(gambler, mces.qsa(), RealScalar.of(.1));
        ExploringStartsBatch.apply(gambler, mces, policyInterface);
      }
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, mces.qsa(), ref)));
    }
    gsw.close();
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    discreteVs.print(ROUND);
  }
}
