// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

class FVPE_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    PolicyInterface policyInterface = VI_Gambler.getOptimalPolicy(gambler);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gambler, policyInterface, //
        gambler, RealScalar.ONE);
    DiscreteVs vs = fvpe.simulate(12030);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.001)));
    Put.of(new File("/home/datahaki/fvmc_gambler"), vs.values());
  }
}
