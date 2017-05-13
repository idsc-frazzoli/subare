// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.Settings;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Put;

class MCES_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    PolicyInterface policyInterface = VI_Gambler.getOptimalPolicy(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        gambler, policyInterface, gambler, RealScalar.ONE, RealScalar.of(.1));
    mces.simulate(10000);
    DiscreteVs discreteVs = DiscreteVs.create(gambler, mces.getQsa());
    discreteVs.print();
    Put.of(new File(Settings.root(), "mces_gambler"), discreteVs.values());
  }
}
