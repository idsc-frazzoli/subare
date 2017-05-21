// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.io.Put;

class MCES_Gambler {
  public static void main(String[] args) throws Exception {
    Gambler gambler = Gambler.createDefault();
    PolicyInterface policyInterface = GamblerHelper.getOptimalPolicy(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        gambler, policyInterface, gambler, RealScalar.ONE, RealScalar.of(.1));
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("gambler_qsa_mces.gif"), 100);
    for (int index = 0; index < 100; ++index) {
      System.out.println(index);
      mces.simulate(250);
      gsw.append(ImageFormat.of(GamblerHelper.render(gambler, mces.qsa())));
    }
    gsw.close();
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    discreteVs.print();
    Put.of(UserHome.file("mces_gambler"), discreteVs.values());
  }
}
