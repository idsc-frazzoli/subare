// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Put;

class MCES_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    PolicyInterface policyInterface = GamblerHelper.getOptimalPolicy(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        gambler, policyInterface, gambler, RealScalar.ONE, RealScalar.of(.1));
    mces.simulate(10000);
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.getQsa());
    discreteVs.print();
    Put.of(UserHome.file("mces_gambler"), discreteVs.values());
  }
}
