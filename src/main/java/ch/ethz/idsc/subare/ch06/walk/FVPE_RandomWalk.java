// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.sca.Round;

class FVPE_RandomWalk {
  public static void main(String[] args) {
    RandomWalk gridWorld = new RandomWalk();
    PolicyInterface policyInterface = new EquiprobablePolicy(gridWorld);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        gridWorld, policyInterface, //
        gridWorld, RealScalar.ONE);
    DiscreteVs vs = DiscreteVs.build(gridWorld);
    vs.setAll(fvpe.simulate(123));
    vs.print(Round.toMultipleOf(DecimalScalar.of(.01)));
  }
}
