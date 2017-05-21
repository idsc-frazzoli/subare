// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.IOException;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

class FVPE_Gambler {
  public static void main(String[] args) throws IOException {
    Gambler gambler = Gambler.createDefault();
    PolicyInterface policyInterface = GamblerHelper.getOptimalPolicy(gambler);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gambler, policyInterface, //
        gambler, RealScalar.ONE);
    DiscreteVs vs = fvpe.simulate(12030);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.001)));
    Put.of(UserHome.file("fvmc_gambler"), vs.values());
  }
}
