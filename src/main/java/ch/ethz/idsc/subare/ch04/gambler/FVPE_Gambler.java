// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Put;
import ch.ethz.idsc.tensor.sca.Round;

/** {0, 0} 0
 * {0, 1} -18.3
 * {0, 2} -27.5
 * {0, 3} -28.3
 * {1, 0} -17.4
 * {1, 1} -22.0
 * {1, 2} -25.0
 * {1, 3} -22.5
 * {2, 0} -22.7
 * {2, 1} -23.7
 * {2, 2} -21.1
 * {2, 3} -16.2
 * {3, 0} -26.1
 * {3, 1} -21.5
 * {3, 2} -13.1
 * {3, 3} 0 */
class FVPE_Gambler {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.001));

  public static void main(String[] args) throws IOException {
    Gambler gambler = new Gambler(100, RationalScalar.of(40, 100));
    PolicyInterface policyInterface = VI_Gambler.getOptimalPolicy(gambler);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gambler, policyInterface, //
        gambler, RealScalar.ONE);
    DiscreteVs vs = fvpe.simulate(502030);
    Tensor result = vs.values();
    Put.of(new File("/home/datahaki/fvmc_gambler"), result);
    Index statesIndex = Index.build(gambler.states());
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println(state + " " + result.get(stateI).map(ROUND)); // .map(ROUND)
    }
  }
}
