// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

/** first visit policy evaluation determines state values v(s)
 * 
 * 0 0
 * 1 0.16
 * 2 0.30
 * 3 0.47
 * 4 0.64
 * 5 0.84
 * 6 0 */
class FVPE_Randomwalk {
  public static void main(String[] args) {
    Randomwalk randomwalk = new Randomwalk();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        randomwalk, new EquiprobablePolicy(randomwalk), //
        randomwalk, RealScalar.ONE, null);
    DiscreteVs vs = fvpe.simulate(1023);
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
